import abc
import collections
import random
from dataclasses import dataclass  # ← new import
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass(slots=True)
class Experience:
    """One-step or n-step transition."""

    state: dict | np.ndarray
    action: int
    reward: float
    next_state: dict | np.ndarray
    done: bool
    n: int = 1  # n-step distance; default 1 for 1-step


class NStepReplayBuffer:
    def __init__(self, capacity: int, n: int, gamma: float):
        self.capacity = capacity
        self.n = n
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=capacity)
        self.n_step_queue = collections.deque(maxlen=n)

    def _calc_n_step(self) -> Experience:
        state, action, _, _, _ = self.n_step_queue[0]
        R, next_state, done = 0.0, None, False
        for idx, (_, _, r, nxt, dn) in enumerate(self.n_step_queue):
            R += (self.gamma ** (idx + 1)) * r  # unchanged
            next_state, done = nxt, dn
            if dn:
                break
        return Experience(state, action, R, next_state, done, idx + 1)

    def push(
        self,
        state_dict: Union[dict, np.ndarray],
        action_idx: int,
        reward: float,
        next_state_dict: Union[dict, np.ndarray],
        done: bool,
    ):
        self.n_step_queue.append(
            (state_dict, action_idx, reward, next_state_dict, done)
        )

        # emit when the window is full
        if len(self.n_step_queue) == self.n:
            self.buffer.append(self._calc_n_step())
            self.n_step_queue.popleft()

        # flush the remainder at episode end
        if done:
            while self.n_step_queue:
                self.buffer.append(self._calc_n_step())
                self.n_step_queue.popleft()

    def sample(
        self, batch_size: int
    ) -> List[
        Tuple[Union[dict, np.ndarray], int, float, Union[dict, np.ndarray], bool, int]
    ]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        state_dict: Union[dict, np.ndarray],
        action_idx: int,
        reward: float,
        next_state_dict: Union[dict, np.ndarray],
        done: bool,
    ):
        """Saves an experience."""
        self.buffer.append(
            Experience(state_dict, action_idx, reward, next_state_dict, done)
        )

    def sample(
        self, batch_size: int
    ) -> List[
        Tuple[Union[dict, np.ndarray], int, float, Union[dict, np.ndarray], bool]
    ]:
        """Randomly samples a batch of experiences from memory."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class BaseAgent(abc.ABC):
    def __init__(self, device: str, *, word_length: int | None = None):
        self.device = torch.device(device)
        self.word_length = word_length  # ← new; can be None

    def _state_to_tensor(
        self, state_dict: Union[dict, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = (
            torch.tensor(state_dict["board_representation"], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        alpha = (
            torch.tensor(state_dict["alphabet_status"], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        turn = (
            torch.tensor(state_dict["current_turn"], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        return board, alpha, turn

    def _batch_states_to_tensor(
        self, dicts: List[Union[dict, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compact batching helper that mirrors the original NumPy-based behaviour
        but avoids object-dtype pitfalls.
        Shapes returned:
            boards : (B, max_turns, word_length, 2)
            alphs  : (B, alphabet_size)
            turns  : (B, 1)
        All tensors are `torch.long` on `self.device`.
        """
        batched_boards = torch.stack(
            [
                torch.as_tensor(d["board_representation"], dtype=torch.long)
                for d in dicts
            ]
        ).to(self.device)

        batched_alphs = torch.stack(
            [torch.as_tensor(d["alphabet_status"], dtype=torch.long) for d in dicts]
        ).to(self.device)

        # 1. Coerce every turn to a *scalar* tensor
        turns_list = [
            torch.as_tensor(d["current_turn"], dtype=torch.long) for d in dicts
        ]
        # 2. Stack → shape (B,)
        batched_turns = torch.stack(turns_list).to(self.device)
        # 3. Add final dim once → shape (B,1)
        if batched_turns.dim() == 1:  # typical case
            batched_turns = batched_turns.unsqueeze(1)
        elif batched_turns.shape[-1] != 1:
            raise ValueError(
                f"current_turn batch has unexpected shape {batched_turns.shape}"
            )

        return batched_boards, batched_alphs, batched_turns

    def compute_reward(
        self,
        feedback: List[str] | None,
        game_over: bool,
        won: bool,
        turns_this: int,
        possible_words: List[str],
        initial_vocab: int,
        divide: bool = False,
    ) -> float:
        """Compute per-letter and terminal rewards, scaling win by inverse turns."""
        if self.word_length is None:
            raise AttributeError("BaseAgent.word_length was not set; reward needs it.")
        if won:
            return 1.0 / turns_this
        if game_over and not won:
            return -1.0
        if feedback is None:
            return -0.5
        info_gain_bonus = -np.log(max(len(possible_words), 1)) / np.log(initial_vocab)
        info_gain_bonus = np.clip(info_gain_bonus, -1.0, 0.0)
        gr = 1.0 / self.word_length
        yr = gr / 2.0
        gc = sum(1 for f in feedback if f == "green")
        yc = sum(1 for f in feedback if f == "yellow")
        if divide:
            return (gc * gr + yc * yr + info_gain_bonus) / turns_this
        else:
            tp = 0.1 * turns_this
            return (gc * gr + yc * yr + info_gain_bonus) - tp

    @abc.abstractmethod
    def select_action(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def store_experience(self, *args, **kwargs):
        pass


class WordleAgentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_length: int,
        max_turns: int,
        num_feedback_types: int,
        alphabet_size: int,
        num_alphabet_states: int,
        embedding_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        output_size: int,
    ):
        super(WordleAgentModel, self).__init__()
        self.word_length = word_length
        self.max_turns = max_turns
        self.embedding_dim = embedding_dim

        # Embeddings
        # Character embedding: vocab_size for actual chars + 1 for padding_idx=0. Inputs will be 0 (padding), 1-vocab_size (chars).
        self.char_embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        # Feedback embedding: num_feedback_types for actual feedback + 1 for padding_idx=0. Inputs will be 0 (padding), 1-num_feedback_types.
        self.feedback_embedding = nn.Embedding(
            num_feedback_types + 1, embedding_dim, padding_idx=0
        )

        # Transformer for board representation
        # Each cell (char, feedback) is embedded, concatenated -> 2 * embedding_dim
        # A turn's representation: word_length * (2 * embedding_dim)
        self.board_transformer_input_dim = word_length * 2 * embedding_dim
        encoder_layer_board = nn.TransformerEncoderLayer(
            d_model=self.board_transformer_input_dim,
            nhead=transformer_heads,
            batch_first=True,
            dim_feedforward=self.board_transformer_input_dim * 4,  # Standard practice
        )
        self.board_transformer = nn.TransformerEncoder(
            encoder_layer_board, num_layers=transformer_layers
        )

        # Alphabet status processing
        # num_alphabet_states are all meaningful (0:unknown, 1:gray, etc.). No padding_idx.
        self.alphabet_status_embedding = nn.Embedding(
            num_alphabet_states, embedding_dim
        )
        self.alphabet_fc = nn.Linear(alphabet_size * embedding_dim, 128)

        # Current turn processing
        self.turn_fc = nn.Linear(1, 32)

        # Combine all features
        # Output of transformer is (batch, max_turns, self.board_transformer_input_dim)
        # We will use mean pooling over non-padded turns.
        self.combined_fc_input_dim = self.board_transformer_input_dim + 128 + 32

        self.fc_shared = nn.Sequential(
            nn.Linear(self.combined_fc_input_dim, 256), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # V(s)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),  # A(s,a)
        )

    def forward(
        self,
        board_representation: torch.Tensor,
        alphabet_status: torch.Tensor,
        current_turn: torch.Tensor,
    ) -> torch.Tensor:
        # board_representation: (batch, max_turns, word_length, 2)
        #   board_representation[..., 0] are char_ids (-1 for padding, 0-25 for letters)
        #   board_representation[..., 1] are feedback_ids (0 for padding, 1-3 for gray,yellow,green)
        # alphabet_status: (batch, 26) (values 0-3)
        # current_turn: (batch, 1)

        batch_size = board_representation.size(0)

        # Process board_representation
        # char_ids: map -1 (padding) to 0, and 0-25 ('a'-'z') to 1-26 for char_embedding
        char_ids = board_representation[:, :, :, 0] + 1
        # feedback_ids: already 0 for padding, 1-N for feedback types
        feedback_ids = board_representation[:, :, :, 1]

        embedded_chars = self.char_embedding(
            char_ids
        )  # (batch, max_turns, word_length, embedding_dim)
        embedded_feedback = self.feedback_embedding(
            feedback_ids
        )  # (batch, max_turns, word_length, embedding_dim)

        board_combined_embeddings = torch.cat(
            (embedded_chars, embedded_feedback), dim=-1
        )
        # (batch, max_turns, word_length, 2 * embedding_dim)

        transformer_input = board_combined_embeddings.reshape(
            batch_size, self.max_turns, -1
        )
        # (batch, max_turns, word_length * 2 * embedding_dim)

        padding_mask = char_ids[:, :, 0] == 0  # (batch, max_turns)
        all_pad_rows = padding_mask.all(dim=1)  # (batch,)

        # Pre-allocate tensor to gather both groups
        transformer_output = torch.zeros(
            batch_size,
            self.max_turns,
            self.board_transformer_input_dim,
            device=transformer_input.device,
            dtype=transformer_input.dtype,
        )

        # 1) Normal rows: at least one True -> False mix, the rest is already zero
        normal_idx = (~all_pad_rows).nonzero(as_tuple=True)[0]
        if normal_idx.numel():  # may be empty
            out = self.board_transformer(
                transformer_input[normal_idx],
                src_key_padding_mask=padding_mask[normal_idx],
            )
            transformer_output[normal_idx] = out

        # Mean pool transformer output, masking out padding.
        # Expand padding_mask for broadcasting to zero out padded positions before summing
        expanded_padding_mask = padding_mask.unsqueeze(-1).expand_as(transformer_output)
        transformer_output_masked = transformer_output.masked_fill(
            expanded_padding_mask, 0.0
        )
        summed_transformer_output = transformer_output_masked.sum(
            dim=1
        )  # Sum over max_turns

        num_non_padded_turns = (
            (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        )  # (batch, 1)
        board_features = (
            summed_transformer_output / num_non_padded_turns
        )  # (batch, self.board_transformer_input_dim)

        # Process alphabet_status
        # alphabet_status: (batch, 26), values are 0-3. Embedding does not use padding_idx.
        embedded_alphabet = self.alphabet_status_embedding(
            alphabet_status
        )  # (batch, 26, embedding_dim)
        flattened_alphabet = embedded_alphabet.view(
            batch_size, -1
        )  # (batch, 26 * embedding_dim)
        alphabet_features = F.relu(self.alphabet_fc(flattened_alphabet))  # (batch, 128)

        # Process current_turn
        turn_features = F.relu(self.turn_fc(current_turn.float()))  # (batch, 32)

        # Combine features
        combined_features = torch.cat(
            (board_features, alphabet_features, turn_features), dim=1
        )

        shared = self.fc_shared(combined_features)  # (batch, 256)

        value = self.value_stream(shared)  # (batch, 1)
        adv = self.adv_stream(shared)  # (batch, |A|)
        q = value + adv - adv.mean(dim=1, keepdim=True)  # dueling combine
        return q


class DQNRLAgent(BaseAgent):
    def __init__(
        self,
        all_words: List[str],
        word_length: int,
        max_turns: int,
        embedding_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_frequency: int = 100,
        n_steps: int = 3,
    ):

        super().__init__(device, word_length=word_length)

        self.word_length = word_length
        self.max_turns = max_turns
        self.all_words = sorted(list(set(all_words)))  # Ensure unique and sorted
        self.word_to_idx = {word: i for i, word in enumerate(self.all_words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.all_words)}
        self.output_size = len(self.all_words)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.train_step_counter = 0

        # Constants for model initialization, derived from game's encoding logic
        self.char_vocab_size = 26  # 'a' through 'z'
        self.num_feedback_types = (
            3  # gray, yellow, green (integer codes 1, 2, 3; 0 is padding)
        )
        self.alphabet_size = 26
        self.num_alphabet_states = (
            4  # unknown, gray, yellow, green (integer codes 0, 1, 2, 3)
        )

        if self.output_size == 0:
            raise ValueError("The 'all_words' list cannot be empty.")
        if (word_length * 2 * embedding_dim) % transformer_heads != 0:
            raise ValueError(
                f"Transformer input dim ({word_length * 2 * embedding_dim}) must be divisible by transformer_heads ({transformer_heads})"
            )

        self.model = WordleAgentModel(
            vocab_size=self.char_vocab_size,
            word_length=self.word_length,
            max_turns=self.max_turns,
            num_feedback_types=self.num_feedback_types,
            alphabet_size=self.alphabet_size,
            num_alphabet_states=self.num_alphabet_states,
            embedding_dim=embedding_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            output_size=self.output_size,
        ).to(self.device)

        self.target_model = WordleAgentModel(
            vocab_size=self.char_vocab_size,
            word_length=self.word_length,
            max_turns=self.max_turns,
            num_feedback_types=self.num_feedback_types,
            alphabet_size=self.alphabet_size,
            num_alphabet_states=self.num_alphabet_states,
            embedding_dim=embedding_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            output_size=self.output_size,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.n_steps = n_steps
        self.replay_buffer = NStepReplayBuffer(
            buffer_capacity, n=self.n_steps, gamma=self.gamma
        )

    @torch.inference_mode()
    def select_action(
        self,
        state_dict: Union[dict, np.ndarray],
        possible_actions_set: set | List[str],
        epsilon: float,
    ) -> str:
        """
        Selects an action word string using epsilon-greedy policy.
        Filters actions based on 'possible_actions_set'.
        """
        if not isinstance(possible_actions_set, set):
            possible_actions_set = set(possible_actions_set)

        valid_words_in_dict = [
            word for word in possible_actions_set if word in self.word_to_idx
        ]

        if not valid_words_in_dict:
            # This can happen if possible_actions_set is empty or contains words not in agent's dictionary
            # Fallback: if possible_actions_set has items, pick one randomly, else error.
            if possible_actions_set:
                # Log a warning here if possible
                # print(f"Warning: No words from possible_actions_set found in agent's dictionary. Choosing randomly from possible_actions_set.")
                return random.choice(list(possible_actions_set))
            else:
                raise ValueError(
                    "No possible actions to select from, and possible_actions_set is empty."
                )

        if random.random() < epsilon:
            action_word = random.choice(valid_words_in_dict)
        else:
            self.model.eval()
            board_tensor, alpha_status_tensor, current_turn_tensor = (
                self._state_to_tensor(state_dict)
            )
            action_scores = self.model(
                board_tensor, alpha_status_tensor, current_turn_tensor
            ).squeeze(0)

            valid_action_indices = [
                self.word_to_idx[word] for word in valid_words_in_dict
            ]

            # Ensure valid_action_indices is not empty before indexing action_scores
            if (
                not valid_action_indices
            ):  # Should be caught by earlier check, but as a safeguard
                return random.choice(valid_words_in_dict)

            # Create a mask for valid actions to apply to scores
            mask = torch.full_like(action_scores, float("-inf"))
            mask[valid_action_indices] = 0.0
            masked_scores = action_scores + mask  # Add -inf to invalid actions

            best_action_idx = torch.argmax(masked_scores).item()
            action_word = self.idx_to_word[best_action_idx]

        return action_word

    def store_experience(
        self,
        state_dict: Union[dict, np.ndarray],
        action_word: str,
        reward: float,
        next_state_dict: Union[dict, np.ndarray],
        done: bool,
    ):
        if action_word not in self.word_to_idx:
            print(
                f"Warning: Action word '{action_word}' not in agent's dictionary. Experience not stored."
            )
            return

        action_idx = self.word_to_idx[action_word]
        self.replay_buffer.push(state_dict, action_idx, reward, next_state_dict, done)

    def learn(self) -> Union[None, float]:
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to learn

        self.model.train()  # Set model to training mode

        experiences = self.replay_buffer.sample(self.batch_size)
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones = [e.done for e in experiences]
        ns = [e.n for e in experiences]

        # Convert to batched tensors
        current_state_tensors = self._batch_states_to_tensor(list(states))
        next_state_tensors = self._batch_states_to_tensor(list(next_states))

        actions_tensor = (
            torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        )
        rewards_tensor = (
            torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        )
        dones_tensor = (
            torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(self.device)
        )  # 0 for not done, 1 for done

        # Get Q-values for current states and actions taken
        # self.model(*current_state_tensors) returns scores for all actions
        current_q_values_all_actions = self.model(*current_state_tensors)
        # Select the Q-value for the action that was actually taken
        current_q_values = current_q_values_all_actions.gather(1, actions_tensor)

        # Get target Q-values from next states using the target network
        with torch.no_grad():
            # 1) action selection by the *online* network
            next_q_online = self.model(*next_state_tensors)  # Qθ(s′,·)
            next_actions_online = next_q_online.argmax(1, keepdim=True)  # (batch,1)

            # 2) action evaluation by the *target* network
            next_q_target = self.target_model(*next_state_tensors)  # Qθ¯(s′,·)
            max_next_q_values_target = next_q_target.gather(
                1, next_actions_online
            )  # (batch,1)

            # 3) TD target
            n_tensor = torch.as_tensor(
                ns, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            gamma_scalar = torch.tensor(self.gamma, device=self.device)
            gamma_tensor = gamma_scalar.pow(n_tensor)

            target_q_values = (
                rewards_tensor
                + gamma_tensor * max_next_q_values_target * (1 - dones_tensor)
            )

        # Compute loss (e.g., Mean Squared Error)
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_frequency == 0:
            self.update_target_network()
        return loss.item()  # Return loss for monitoring

    def update_target_network(self) -> None:
        """
        Hard-copy the online network parameters to the target network.
        Call this every `self.target_update_frequency` training steps.
        """
        self.target_model.load_state_dict(self.model.state_dict())


# Q network
class QRLAgent(BaseAgent):
    """Basic Q-learning agent with experience replay (no target network)."""

    def __init__(
        self,
        all_words: List[str],
        word_length: int,
        max_turns: int,
        embedding_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__(device, word_length=word_length)
        self.gamma = gamma
        self.word_length = word_length
        self.max_turns = max_turns
        # Vocabulary mapping
        self.all_words = sorted(set(all_words))
        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.all_words)}
        self.output_size = len(self.all_words)
        # Q-network
        self.char_vocab_size = 26
        self.num_feedback_types = 3
        self.alphabet_size = 26
        self.num_alphabet_states = 4
        self.model = WordleAgentModel(
            vocab_size=self.char_vocab_size,
            word_length=word_length,
            max_turns=max_turns,
            num_feedback_types=self.num_feedback_types,
            alphabet_size=self.alphabet_size,
            num_alphabet_states=self.num_alphabet_states,
            embedding_dim=embedding_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            output_size=self.output_size,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    @torch.inference_mode()
    def select_action(
        self,
        state_dict: Union[dict, np.ndarray],
        possible_actions_set: Union[set, List[str]],
        epsilon: float,
    ) -> str:
        """Epsilon-greedy action selection."""
        if not isinstance(possible_actions_set, set):
            possible_actions_set = set(possible_actions_set)
        valid_words = [w for w in possible_actions_set if w in self.word_to_idx]
        if not valid_words:
            return random.choice(list(possible_actions_set))
        if random.random() < epsilon:
            return random.choice(valid_words)
        self.model.eval()
        board_t, alpha_t, turn_t = self._state_to_tensor(state_dict)
        scores = self.model(board_t, alpha_t, turn_t).squeeze(0)
        mask = torch.full_like(scores, float("-inf"))
        idxs = [self.word_to_idx[w] for w in valid_words]
        mask[idxs] = 0.0
        gated = scores + mask
        best_idx = torch.argmax(gated).item()
        return self.idx_to_word[best_idx]

    def store_experience(
        self,
        state_dict: Union[dict, np.ndarray],
        action_word: str,
        reward: float,
        next_state_dict: Union[dict, np.ndarray],
        done: bool,
    ):
        """Store transition in replay buffer."""
        if action_word not in self.word_to_idx:
            return
        idx = self.word_to_idx[action_word]
        self.replay_buffer.push(state_dict, idx, reward, next_state_dict, done)

    def learn(self) -> Union[None, float]:
        """Sample batch and perform Q-learning update."""
        if len(self.replay_buffer) < self.batch_size:
            return
        self.model.train()
        exp = self.replay_buffer.sample(self.batch_size)
        states = [e.state for e in exp]
        actions = [e.action for e in exp]
        rewards = [e.reward for e in exp]
        next_states = [e.next_state for e in exp]
        dones = [e.done for e in exp]

        bs = list(states)
        nb = list(next_states)
        cb, ca, ct = self._batch_states_to_tensor(bs)
        nb_b, na_b, nt_b = self._batch_states_to_tensor(nb)
        act_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rew_t = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(
            1
        )
        done_t = torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
        qv = self.model(cb, ca, ct).gather(1, act_t)
        with torch.no_grad():
            nq = self.model(nb_b, na_b, nt_b).max(1)[0].unsqueeze(1)
            tgt = rew_t + self.gamma * nq * (1 - done_t)
        # loss = F.mse_loss(qv, tgt)
        loss = F.smooth_l1_loss(qv, tgt)  # Huber loss can be used as well
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# Random guessing agent for benchmarking
class RandomAgent(BaseAgent):
    """Random guessing agent baseline."""

    def __init__(
        self,
        all_words: List[str],
        word_length: int,
        max_turns: int,
        device: str = "cpu",
    ):
        super().__init__(device)
        self.all_words = list(all_words)
        self.word_length = word_length
        self.max_turns = max_turns

    def select_action(
        self,
        state_dict: Union[dict, np.ndarray],
        possible_actions_set: Union[set, List[str]],
        epsilon: float = 0.0,
    ) -> str:
        # Always choose randomly from possible actions
        candidates = list(possible_actions_set)
        return random.choice(candidates)

    def store_experience(self, *args, **kwargs):
        pass

    def learn(self):
        pass
