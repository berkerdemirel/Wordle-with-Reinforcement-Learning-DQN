import nltk
import torch
from nltk.corpus import words as nltk_words
from omegaconf import DictConfig

from agent import DQNRLAgent, QRLAgent, RandomAgent


def load_nltk_words(word_length):
    """Loads words from NLTK, downloads corpus if necessary."""
    try:
        # Check if 'words' corpus is available
        nltk_words.ensure_loaded()
    except LookupError:
        print("NLTK 'words' corpus not found. Downloading...")
        nltk.download("words", quiet=True)
        # Re-initialize after download
        # This re-import is crucial because the module's state might not update otherwise
        from nltk.corpus import words as reloaded_nltk_words

        all_english_words = reloaded_nltk_words.words()
    else:
        all_english_words = nltk_words.words()

    # Filter for words of the specified length and ensure they are all alphabetic and lowercase
    filtered_words = [
        word.lower()
        for word in all_english_words
        if len(word) == word_length and word.isalpha()
    ]
    filtered_words_list = list(set(filtered_words))
    if not filtered_words_list:
        print(f"No words of length {word_length} found in NLTK corpus. Exiting.")
        exit()

    return filtered_words_list  # Use set to remove duplicates, then convert to list


def create_agent(cfg: DictConfig, vocab: list, device: torch.device):
    if cfg.get("agent", "dqn") == "dqn":
        agent = DQNRLAgent(
            all_words=vocab,
            word_length=cfg.game.word_length,
            max_turns=cfg.game.max_turns,
            gamma=cfg.rl.gamma,
            buffer_capacity=cfg.train.replay_capacity,
            batch_size=cfg.train.batch_size,
            n_steps=cfg.rl.n_step,
            device=device,
            learning_rate=cfg.optim.lr_initial,  # temporary, replaced below
        )
    elif cfg.get("agent", "qn") == "qn":
        agent = QRLAgent(
            all_words=vocab,
            word_length=cfg.game.word_length,
            max_turns=cfg.game.max_turns,
            gamma=cfg.rl.gamma,
            buffer_capacity=cfg.train.replay_capacity,
            batch_size=cfg.train.batch_size,
            device=device,
            learning_rate=cfg.optim.lr_initial,
        )
    elif cfg.get("agent") == "random":
        agent = RandomAgent(
            all_words=vocab,
            word_length=cfg.game.word_length,
            max_turns=cfg.game.max_turns,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {cfg.get('agent')}")
    return agent


def build_optimizer(model, cfg_opt):
    """Return (optimizer, scheduler_or_None) given model and cfg subsections."""
    opt_cls = getattr(torch.optim, cfg_opt.optimizer)
    optimizer = opt_cls(
        model.parameters(),
        lr=cfg_opt.lr_initial,
        weight_decay=cfg_opt.get("weight_decay", 0.0),
    )
    scheduler = None
    if cfg_opt.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg_opt.scheduler_T
        )
    return optimizer, scheduler


def linear_epsilon(cfg: DictConfig, step: int) -> float:
    eps0, eps1, decay = (
        cfg.exploration.epsilon_start,
        cfg.exploration.epsilon_end,
        cfg.exploration.decay_steps,
    )
    step = min(step, decay)
    return eps0 + (eps1 - eps0) * step / decay
