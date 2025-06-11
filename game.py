import random
from typing import Dict, Tuple

import numpy as np  # Added for numerical state representation
from utils import load_nltk_words  # Import the function from utils.py


class WordleGame:
    def __init__(
        self,
        word_length: int = 5,
        max_turns: int = 6,
        all_words: list = None,
    ):
        # Load or accept custom word list
        if all_words is None:
            from utils import load_nltk_words

            all_words = load_nltk_words(word_length)

        self.word_length = word_length
        # Create a list of lowercased words of the correct length first
        initial_possible_words = [
            word.lower()
            for word in all_words
            if len(word) == self.word_length and word.isalpha()
        ]

        if not initial_possible_words:
            raise ValueError(
                f"No words of length {self.word_length} found in the provided word list."
            )
        self.initial_vocab = len(initial_possible_words)
        self.secret_word = random.choice(initial_possible_words)
        # Convert to a set for fast membership testing
        self.possible_words = set(initial_possible_words)

        self.max_turns = max_turns
        self.current_turn = 0
        self.guesses_history = []  # To store (guess_word_str, feedback_list) tuples

        # Alphabet and feedback constants
        self.CORRECT_POSITION = "green"
        self.CORRECT_LETTER_WRONG_POSITION = "yellow"
        self.INCORRECT_LETTER = "gray"
        self.STATUS_UNKNOWN = "unknown"

        self.alphabet_status = {
            chr(ord("a") + i): self.STATUS_UNKNOWN for i in range(26)
        }

        # Mappings for state encoding
        self.char_to_int = {chr(ord("a") + i): i for i in range(26)}
        self.int_to_char = {
            i: chr(ord("a") + i) for i in range(26)
        }  # For potential debugging or agent output

        self.feedback_to_int = {
            # 0 is reserved for "no feedback" / padding
            self.INCORRECT_LETTER: 1,
            self.CORRECT_LETTER_WRONG_POSITION: 2,
            self.CORRECT_POSITION: 3,
        }
        self.alphabet_status_to_int = {
            self.STATUS_UNKNOWN: 0,
            self.INCORRECT_LETTER: 1,
            self.CORRECT_LETTER_WRONG_POSITION: 2,
            self.CORRECT_POSITION: 3,
        }

    def _evaluate_guess(self, guess: str) -> list:
        """
        Evaluate a guess against the secret word and return feedback.
        Feedback is a list of length word_length with:
        - "green" for correct letters in the correct position,
        - "yellow" for correct letters in the wrong position,
        - "gray" for incorrect letters.
        """
        guess = guess.lower()
        feedback = [self.INCORRECT_LETTER] * self.word_length
        secret_word_letter_counts = {}
        for letter in self.secret_word:
            secret_word_letter_counts[letter] = (
                secret_word_letter_counts.get(letter, 0) + 1
            )

        # First pass for green letters
        for i in range(self.word_length):
            if guess[i] == self.secret_word[i]:
                feedback[i] = self.CORRECT_POSITION
                secret_word_letter_counts[guess[i]] -= 1
                # Update alphabet_status: Green is highest priority
                self.alphabet_status[guess[i]] = self.CORRECT_POSITION

        # Second pass for yellow letters
        for i in range(self.word_length):
            if feedback[i] == self.CORRECT_POSITION:  # Already processed as green
                continue
            if (
                guess[i] in self.secret_word
                and secret_word_letter_counts.get(guess[i], 0) > 0
            ):
                feedback[i] = self.CORRECT_LETTER_WRONG_POSITION
                secret_word_letter_counts[guess[i]] -= 1
                # Update alphabet_status: Yellow if not already green
                if self.alphabet_status[guess[i]] != self.CORRECT_POSITION:
                    self.alphabet_status[guess[i]] = self.CORRECT_LETTER_WRONG_POSITION
            else:  # Letter is not in word or all instances accounted for
                # Update alphabet_status: Gray if unknown (lowest priority)
                if self.alphabet_status[guess[i]] == self.STATUS_UNKNOWN:
                    self.alphabet_status[guess[i]] = self.INCORRECT_LETTER
        return feedback

    def _feedback_for(self, guess: str, secret: str) -> list:
        """Return feedback list for guess against given secret word."""
        feedback = [self.INCORRECT_LETTER] * self.word_length
        counts = {}
        for ch in secret:
            counts[ch] = counts.get(ch, 0) + 1
        # greens
        for i, ch in enumerate(guess):
            if ch == secret[i]:
                feedback[i] = self.CORRECT_POSITION
                counts[ch] -= 1
        # yellows
        for i, ch in enumerate(guess):
            if feedback[i] == self.CORRECT_POSITION:
                continue
            if counts.get(ch, 0) > 0:
                feedback[i] = self.CORRECT_LETTER_WRONG_POSITION
                counts[ch] -= 1
        return feedback

    # Update _filter_possible_words to enforce manual green/yellow/gray rules with duplicate handling
    def _filter_possible_words(self, guess: str, feedback: str):
        """Prune possible_words to those consistent with the last feedback, including duplicate counts."""
        self.possible_words = {
            candidate
            for candidate in self.possible_words
            if self._feedback_for(guess, candidate) == feedback
        }

    def make_guess(self, guess_word: str) -> Tuple[str, list, bool]:
        """
        Process a guess word, evaluate it against the secret word,
        and update the game state accordingly.
        Returns a tuple of:
        - A message indicating the result of the guess.
        - A list of feedback for each letter in the guess.
        - A boolean indicating if the game is over (won or lost).
        """
        if len(guess_word) != self.word_length:
            return "Invalid guess length.", None, False  # Message, feedback, game_over

        guess_word = guess_word.lower()

        if guess_word not in self.possible_words:
            return (
                "Not a valid word in the dictionary.",
                None,
                False,
            )  # Message, feedback, game_over

        self.current_turn += 1
        feedback = self._evaluate_guess(guess_word)
        self.guesses_history.append((guess_word, feedback))
        # Prune future possible words to remain consistent
        self._filter_possible_words(guess_word, feedback)

        won = all(f == self.CORRECT_POSITION for f in feedback)
        lost = self.current_turn >= self.max_turns and not won
        game_over = won or lost

        if won:
            return "Congratulations! You guessed the word.", feedback, game_over
        if lost:
            return f"Game Over. The word was {self.secret_word}.", feedback, game_over

        return "Guess processed.", feedback, game_over

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the game as a dictionary of numpy arrays.
        The state includes:
        - A board representation of guesses and feedback.
        - An encoded representation of the alphabet status.
        - The current turn number."""
        # Board representation: (max_turns, word_length, 2)
        # Layer 0: character int (-1 for padding)
        # Layer 1: feedback int (0 for padding/no feedback)
        board_representation = np.full(
            (self.max_turns, self.word_length, 2), fill_value=0, dtype=np.int32
        )
        board_representation[:, :, 0] = -1  # Initialize char layer with -1 for padding

        for turn_idx, (guess_str, feedback_list) in enumerate(self.guesses_history):
            if (
                turn_idx < self.max_turns
            ):  # Should always be true if history is managed well
                for char_idx in range(self.word_length):
                    if char_idx < len(guess_str):  # Should always be true
                        board_representation[turn_idx, char_idx, 0] = (
                            self.char_to_int.get(guess_str[char_idx], -1)
                        )
                        board_representation[turn_idx, char_idx, 1] = (
                            self.feedback_to_int.get(feedback_list[char_idx], 0)
                        )

        # Alphabet status encoded: (26,)
        alphabet_status_encoded = np.zeros(26, dtype=np.int32)
        for i in range(26):
            char = self.int_to_char[i]
            alphabet_status_encoded[i] = self.alphabet_status_to_int.get(
                self.alphabet_status[char], 0
            )

        # Current turn encoded: (1,)
        current_turn_encoded = np.array([self.current_turn], dtype=np.int32)

        return {
            "board_representation": board_representation,
            "alphabet_status": alphabet_status_encoded,
            "current_turn": current_turn_encoded,
        }

    def is_game_over(self) -> bool:
        """Check if the game is over based on the last guess and current turn."""
        if not self.guesses_history:
            return False
        last_guess, last_feedback = self.guesses_history[-1]
        won = all(f == self.CORRECT_POSITION for f in last_feedback)
        return won or self.current_turn >= self.max_turns
