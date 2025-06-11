# pl_wordle.py
import random
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from game import WordleGame
from utils import build_optimizer, create_agent, linear_epsilon, load_nltk_words


class WordleLightning(pl.LightningModule):
    """
    A generic RL wrapper that delegates all game / network specifics
    to a `BaseAgent` instance (DQN, QN, or Random).
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # ----- environment & vocab ------------------------------------
        self.vocab: List[str] = load_nltk_words(cfg.game.word_length)
        self._make_fresh_game()

        # ----- agent ---------------------------------------------------
        self.agent = create_agent(cfg, self.vocab, self.device.type)

        # attach optimizer only if the agent has a model
        if hasattr(self.agent, "model"):
            self.opt, self.scheduler = build_optimizer(self.agent.model, cfg.optim)
        else:
            self.opt, self.scheduler = None, None

        # logging helpers
        self._reset_episode_counters()
        self._reset_aggregate_counters()

        # We run our own .backward / .step
        self.automatic_optimization = False
        self.env_step = 0

    def _reset_aggregate_counters(self):
        self.acc_reward = 0.0
        self.acc_turns = 0
        self.acc_loss = 0.0
        self.acc_loss_steps = 0  # number of learner updates
        self.acc_episodes = 0

    # ------------------------------------------------------------------ #
    # Lightning boilerplate                                              #
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        if not hasattr(self.agent, "model"):
            raise RuntimeError("RandomAgent has no forward pass.")
        return self.agent.model(*args, **kwargs)

    def configure_optimizers(self) -> Optional[Dict[str, Any]]:
        if self.opt is None:  # RandomAgent case
            return None
        return {
            "optimizer": self.opt,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        # Dummy DataLoader: Lightning requires one but we manage all work in training_step
        return torch.utils.data.DataLoader(range(1), batch_size=1)

    # ------------------------------------------------------------------ #
    # RL loop: one env step per call                                     #
    # ------------------------------------------------------------------ #
    def training_step(self, *_):
        if self.game.done:
            self._log_episode()
            self._make_fresh_game()

        # ε-greedy schedule only for agents that support it
        eps = (
            linear_epsilon(self.hparams, self.env_step)
            if hasattr(self.agent, "learn")
            else 0.0
        )

        state = self.game.get_state()
        action = self.agent.select_action(state, self.game.possible_words, eps)

        msg, fb, done = self.game.make_guess(action)
        next_state = self.game.get_state()

        reward = self.agent.compute_reward(
            fb,
            done,
            msg.startswith("Congratulations"),
            self.ep_turns + 1,
            possible_words=self.game.possible_words,
            initial_vocab=self.game.initial_vocab,
        )

        # store + (maybe) learn
        if hasattr(self.agent, "store_experience"):
            self.agent.store_experience(state, action, reward, next_state, done)

        loss = None
        if (
            hasattr(self.agent, "learn")
            and self.env_step % self.hparams.train.update_every == 0
        ):
            loss = self.agent.learn()
            if loss is not None:
                # ---- accumulate loss -------------------------------
                self.acc_loss += loss
                self.acc_loss_steps += 1
        # bookkeeping
        self.ep_reward += reward
        self.ep_turns += 1
        if done:
            self.game.done = True

        # ---- periodically flush aggregated metrics -------------------
        if self.env_step > 0 and self.env_step % self.hparams.train.log_every == 0:
            if self.acc_episodes:  # avoid div-by-zero
                avg_turns = self.acc_turns / self.acc_episodes
                avg_reward = self.acc_reward / self.acc_episodes
            else:
                avg_turns = avg_reward = float("nan")

            if self.acc_loss_steps:
                avg_loss = self.acc_loss / self.acc_loss_steps
            else:
                avg_loss = float("nan")
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log(  # same keys, any others you like
                    {
                        "avg_turns": avg_turns,
                        "avg_reward": avg_reward,
                        "avg_loss": avg_loss,
                        "episodes": self.acc_episodes,
                    },
                    step=self.env_step,  # ← use env_step, not global_step
                    commit=True,
                )
            self._reset_aggregate_counters()
        self.env_step += 1

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _make_fresh_game(self):
        self.game = WordleGame(
            self.hparams.game.word_length,
            self.hparams.game.max_turns,
            all_words=self.vocab,
        )
        self.game.possible_words = set(self.vocab)
        if self.game.secret_word not in self.vocab:
            self.game.secret_word = random.choice(self.vocab)
        self.game.done = False
        self._reset_episode_counters()

    def _reset_episode_counters(self):
        self.ep_reward = 0.0
        self.ep_turns = 0
        self.episodes = getattr(self, "episodes", 0)

    def _log_episode(self):
        self.episodes += 1

        # ---- add to aggregate pool -------------------------------
        self.acc_reward += self.ep_reward
        self.acc_turns += self.ep_turns
        self.acc_episodes += 1
