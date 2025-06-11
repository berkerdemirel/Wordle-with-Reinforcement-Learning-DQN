# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from pl_wordle import WordleLightning


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    model = WordleLightning(cfg)
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    wandb_logger = WandbLogger(
        project="wordle-universal",
        config=flat_cfg,  # ← JSON-friendly
    )

    trainer = Trainer(
        max_steps=cfg.train.num_env_steps,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.log_every,
        deterministic=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()

"""
import random

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from game import WordleGame
from utils import build_optimizer, create_agent, linear_epsilon, load_nltk_words


# ------------------------------------------------------------------ #
# main                                                               #
# ------------------------------------------------------------------ #
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # vocabulary
    all_words = load_nltk_words(cfg.game.word_length)
    vocab = all_words  # truncate here if desired

    # ------------------------------------------------------------------
    # initialise agent
    # ------------------------------------------------------------------
    agent = create_agent(cfg, vocab, device)

    # Only for learning agents
    if hasattr(agent, "model"):
        agent.optimizer, agent.scheduler = build_optimizer(agent.model, cfg.optim)
    else:
        agent.optimizer, agent.scheduler = None, None

    # ------------------------------------------------------------------
    # training loop / evaluation for all agents
    # ------------------------------------------------------------------
    total_loss, loss, total_turns, total_rewards, step_ct = 0.0, None, 0, 0.0, 0
    for ep in range(1, cfg.train.num_episodes + 1):
        game = WordleGame(cfg.game.word_length, cfg.game.max_turns, all_words=vocab)
        game.possible_words = set(vocab)
        if game.secret_word not in vocab:
            game.secret_word = random.choice(vocab)

        state = game.get_state()
        ep_reward, turns, won = 0.0, 0, False

        for _ in range(cfg.game.max_turns):
            turns += 1
            step_ct += 1

            eps = linear_epsilon(cfg, step_ct) if cfg.agent in ["dqn", "qn"] else 0.0
            action = agent.select_action(state, game.possible_words, epsilon=eps)

            msg, fb, done = game.make_guess(action)
            next_state = game.get_state()
            reward = agent.compute_reward(
                fb,
                done,
                msg.startswith("Congratulations"),
                turns,
                possible_words=game.possible_words,
                initial_vocab=game.initial_vocab,
            )
            ep_reward += reward

            # Learning-specific operations
            if cfg.agent in ["dqn", "qn"]:
                agent.store_experience(state, action, reward, next_state, done)
                # Learner updates (DQN specific as per original code)
                if cfg.agent == "dqn" and step_ct % cfg.train.update_every == 0:
                    # perform learning step
                    loss = agent.learn()
                    # scheduler step after learning
                    if agent.scheduler:
                        agent.scheduler.step()
                # Note: QN agent's learn() method is not called here based on original structure

            state = next_state
            if done:
                won = msg.startswith("Congratulations")
                break

        total_turns += turns if won else cfg.game.max_turns + 1
        total_rewards += ep_reward
        total_loss += loss if loss is not None else 0.0

        if ep % cfg.train.log_every == 0:
            lr = 0.0
            if hasattr(agent, "optimizer") and agent.optimizer:
                lr = agent.optimizer.param_groups[0]["lr"]

            print(
                f"Ep {ep:5d}/{cfg.train.num_episodes:5d} | ε={eps:.3f} "
                f"| avg_turns={total_turns/ep:.2f} "
                f"| avg_R={total_rewards/ep:.3f} "
                f"| loss={total_loss/ep:.3f} "
                f"| lr={lr:.2e}"
            )

    # ------------------------------------------------------------------
    # testing
    # ------------------------------------------------------------------
    print("\\nTesting agent")
    for k in range(cfg.test.episodes):
        game = WordleGame(cfg.game.word_length, cfg.game.max_turns, all_words=vocab)
        game.possible_words = set(vocab)
        if game.secret_word not in vocab:
            game.secret_word = random.choice(vocab)
        state = game.get_state()
        print(f"\nTest {k+1}: secret={game.secret_word}")
        for _ in range(cfg.game.max_turns):
            guess = agent.select_action(state, game.possible_words, epsilon=0.0)
            msg, fb, done = game.make_guess(guess)
            print(f"  {guess} -> {fb}")
            state = game.get_state()
            if done:
                break
        print(" ", msg)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
"""
