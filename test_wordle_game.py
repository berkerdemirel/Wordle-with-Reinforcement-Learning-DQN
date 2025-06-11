import random
import unittest

import numpy as np  # Import numpy

from game import WordleGame  # Corrected import


class TestWordleGame(unittest.TestCase):

    def setUp(self):
        # Initialize with a fixed secret word for predictable state tests
        self.game_5_letters = WordleGame(word_length=5, max_turns=6)
        self.game_5_letters.secret_word = "apple"  # Fix secret word for some tests

        self.game_6_letters = WordleGame(
            word_length=6,
            max_turns=6,
        )
        self.game_6_letters.secret_word = "banana"

    def test_initialization_valid(self):
        self.assertEqual(self.game_5_letters.word_length, 5)
        self.assertEqual(self.game_5_letters.max_turns, 6)
        self.assertEqual(self.game_5_letters.current_turn, 0)
        self.assertTrue(
            self.game_5_letters.secret_word in self.game_5_letters.possible_words
        )
        self.assertEqual(len(self.game_5_letters.secret_word), 5)

    def test_initialization_custom_turns_length(self):
        game = WordleGame(word_length=6, max_turns=4)
        self.assertEqual(game.word_length, 6)
        self.assertEqual(game.max_turns, 4)

    def test_evaluate_guess_all_green(self):
        self.game_5_letters.secret_word = "apple"
        # Reset alphabet status for this specific test scenario if needed, or ensure setUp does it.
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        feedback = self.game_5_letters._evaluate_guess("apple")
        self.assertEqual(feedback, ["green", "green", "green", "green", "green"])

    def test_evaluate_guess_all_gray(self):
        self.game_5_letters.secret_word = "apple"
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        feedback = self.game_5_letters._evaluate_guess("boing")  # No common letters
        self.assertEqual(feedback, ["gray", "gray", "gray", "gray", "gray"])

    def test_evaluate_guess_mixed_feedback(self):
        self.game_5_letters.secret_word = "slate"
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        # s l a t e
        # c r a n e
        # g y g y g
        feedback = self.game_5_letters._evaluate_guess("crane")
        self.assertEqual(feedback, ["gray", "gray", "green", "gray", "green"])

        self.game_5_letters.secret_word = "audio"
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        # a u d i o
        # a d i e u -> a green, d yellow, i yellow, u yellow
        feedback = self.game_5_letters._evaluate_guess("adieu")
        self.assertEqual(feedback, ["green", "yellow", "yellow", "gray", "yellow"])

    def test_evaluate_guess_duplicate_letters_in_guess_fewer_in_secret(self):
        self.game_5_letters.secret_word = "apple"  # one 'p'
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        # a p p l e
        # p o p p y -> first p yellow, o gray, second p green, third p gray, y gray
        feedback = self.game_5_letters._evaluate_guess("poppy")
        self.assertEqual(feedback, ["yellow", "gray", "green", "gray", "gray"])

        self.game_5_letters.secret_word = "array"  # two 'a's, one 'r'
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }  # Reset for this part
        # a r r a y
        # radar -> r: yellow, a: yellow, d: gray, a: green, r: yellow
        feedback = self.game_5_letters._evaluate_guess("radar")
        self.assertEqual(feedback, ["yellow", "yellow", "gray", "green", "yellow"])

    def test_evaluate_guess_duplicate_letters_in_secret_fewer_in_guess(self):
        self.game_5_letters.secret_word = "geese"  # two 'e's
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }
        # g e e s e
        # elite -> e green, l gray, i gray, t gray, e green
        feedback = self.game_5_letters._evaluate_guess("elite")
        self.assertEqual(feedback, ["yellow", "gray", "gray", "gray", "green"])

    def test_evaluate_guess_duplicate_letters_in_both_exact_match(self):
        self.game_5_letters.secret_word = "apple"
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }  # Reset
        feedback = self.game_5_letters._evaluate_guess("apple")
        self.assertEqual(feedback, ["green", "green", "green", "green", "green"])

    def test_evaluate_guess_duplicate_letters_in_both_partial_match(self):
        # self.game_6_letters.secret_word = "banana" # already set in setUp
        self.game_6_letters.alphabet_status = {
            chr(ord("a") + i): self.game_6_letters.STATUS_UNKNOWN for i in range(26)
        }
        # b a n a n a
        # a n n a t e -> a yellow, n yellow, n green, a green, t gray, e gray
        feedback = self.game_6_letters._evaluate_guess("annate")
        self.assertEqual(
            feedback, ["yellow", "yellow", "green", "green", "gray", "gray"]
        )

    def test_evaluate_guess_case_insensitivity(self):
        self.game_5_letters.secret_word = "apple"
        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }  # Reset for first call
        feedback = self.game_5_letters._evaluate_guess("APPLE")
        self.assertEqual(feedback, ["green", "green", "green", "green", "green"])

        self.game_5_letters.alphabet_status = {
            chr(ord("a") + i): self.game_5_letters.STATUS_UNKNOWN for i in range(26)
        }  # Reset for second call
        feedback = self.game_5_letters._evaluate_guess("ApPlE")
        self.assertEqual(feedback, ["green", "green", "green", "green", "green"])

    def test_make_guess_invalid_length(self):
        message, feedback, game_over = self.game_5_letters.make_guess("four")
        self.assertEqual(message, "Invalid guess length.")
        self.assertIsNone(feedback)
        self.assertFalse(game_over)
        self.assertEqual(self.game_5_letters.current_turn, 0)  # Turn should not advance

    def test_make_guess_win(self):
        self.game_5_letters.secret_word = "crane"
        message, feedback, game_over = self.game_5_letters.make_guess("crane")
        self.assertEqual(message, "Congratulations! You guessed the word.")
        self.assertEqual(feedback, ["green", "green", "green", "green", "green"])
        self.assertTrue(game_over)
        self.assertEqual(self.game_5_letters.current_turn, 1)
        self.assertEqual(self.game_5_letters.guesses_history, [("crane", feedback)])

    def test_make_guess_lose(self):
        self.game_5_letters.secret_word = "aaaaa"
        # Make max_turns incorrect guesses
        guesses = ["flash", "igara", "adapa", "ajaja", "anana", "abama"]
        for i in range(self.game_5_letters.max_turns):
            message, feedback, game_over = self.game_5_letters.make_guess(
                guesses[i]
            )  # A known incorrect word
            if i == self.game_5_letters.max_turns - 1:  # Last guess
                self.assertTrue(game_over)
                self.assertEqual(
                    message,
                    f"Game Over. The word was {self.game_5_letters.secret_word}.",
                )
            else:
                self.assertFalse(game_over)

        self.assertEqual(
            self.game_5_letters.current_turn, self.game_5_letters.max_turns
        )

    def test_make_guess_continue_game(self):
        self.game_5_letters.secret_word = "table"
        message, feedback, game_over = self.game_5_letters.make_guess("chair")
        self.assertEqual(message, "Guess processed.")
        self.assertNotEqual(feedback, ["green", "green", "green", "green", "green"])
        self.assertFalse(game_over)
        self.assertEqual(self.game_5_letters.current_turn, 1)
        self.assertEqual(self.game_5_letters.guesses_history, [("chair", feedback)])

    def test_make_guess_rejects_word_not_in_dictionary(self):
        """Tests that a guess not in self.possible_words is rejected."""
        self.game_5_letters.secret_word = (
            "apple"  # Known secret word for predictable feedback
        )
        guess_not_in_list = "qwert"

        # Ensure the word is not in the initial list for sanity
        self.assertNotIn(guess_not_in_list, self.game_5_letters.possible_words)

        initial_turn = self.game_5_letters.current_turn
        initial_history_len = len(self.game_5_letters.guesses_history)

        message, feedback, game_over = self.game_5_letters.make_guess(guess_not_in_list)

        self.assertEqual(message, "Not a valid word in the dictionary.")
        self.assertIsNone(feedback)
        self.assertFalse(game_over)
        self.assertEqual(
            self.game_5_letters.current_turn, initial_turn
        )  # Turn should not advance
        self.assertEqual(
            len(self.game_5_letters.guesses_history), initial_history_len
        )  # History should not change

    def test_is_game_over(self):
        self.assertFalse(self.game_5_letters.is_game_over())  # Initially false

        self.game_5_letters.secret_word = "apple"
        self.game_5_letters.make_guess("apple")  # Win
        self.assertTrue(self.game_5_letters.is_game_over())
        # Reset for loss condition
        self.setUp()  # Re-initialize game
        self.game_5_letters.secret_word = "aaaaa"
        random.seed(42)  # For reproducibility
        for _ in range(self.game_5_letters.max_turns):
            while True:
                guess = random.choice(list(self.game_5_letters.possible_words))
                if guess != self.game_5_letters.secret_word:
                    break
            self.game_5_letters.make_guess(guess)
        self.assertTrue(self.game_5_letters.is_game_over())

    def test_get_state_structure(self):
        # Test with a fresh game instance or reset state if necessary
        game = WordleGame(word_length=5, max_turns=6)
        game.secret_word = "tests"
        game.make_guess("toast")  # Make one guess
        state = game.get_state()

        self.assertIsInstance(state, dict)
        self.assertIn("board_representation", state)
        self.assertIn("alphabet_status", state)
        self.assertIn("current_turn", state)

        # Board representation checks
        board = state["board_representation"]
        self.assertIsInstance(board, np.ndarray)
        self.assertEqual(board.dtype, np.int32)
        self.assertEqual(board.shape, (game.max_turns, game.word_length, 2))

        # Alphabet status checks
        alpha_status = state["alphabet_status"]
        self.assertIsInstance(alpha_status, np.ndarray)
        self.assertEqual(alpha_status.dtype, np.int32)
        self.assertEqual(alpha_status.shape, (26,))

        # Current turn checks
        current_turn_encoded = state["current_turn"]
        self.assertIsInstance(current_turn_encoded, np.ndarray)
        self.assertEqual(current_turn_encoded.dtype, np.int32)
        self.assertEqual(current_turn_encoded.shape, (1,))
        self.assertEqual(current_turn_encoded[0], 1)  # After one guess

    def test_get_state_encoding_logic(self):
        game = WordleGame(word_length=5, max_turns=6)
        game.secret_word = "apple"  # Known secret word

        # Initial state (Turn 0)
        state_initial = game.get_state()
        board_initial = state_initial["board_representation"]
        # All char encodings should be -1 (padding), feedback 0 (padding)
        self.assertTrue(np.all(board_initial[:, :, 0] == -1))
        self.assertTrue(np.all(board_initial[:, :, 1] == 0))
        # All alphabet status should be 0 (unknown)
        self.assertTrue(
            np.all(
                state_initial["alphabet_status"]
                == game.alphabet_status_to_int[game.STATUS_UNKNOWN]
            )
        )
        self.assertEqual(state_initial["current_turn"][0], 0)

        # Make a guess: "audio" (secret: "apple")
        # Feedback: a:green, u:gray, d:gray, i:gray, o:gray
        # Expected alphabet_status: a:green, u:gray, d:gray, i:gray, o:gray, rest unknown
        game.make_guess("audio")
        state_after_guess1 = game.get_state()
        board1 = state_after_guess1["board_representation"]
        alpha_status1 = state_after_guess1["alphabet_status"]
        current_turn1 = state_after_guess1["current_turn"]

        self.assertEqual(current_turn1[0], 1)

        # Check board_representation for the first guess (turn_idx = 0)
        expected_guess_chars_encoded = [
            game.char_to_int["a"],
            game.char_to_int["u"],
            game.char_to_int["d"],
            game.char_to_int["i"],
            game.char_to_int["o"],
        ]
        expected_guess_feedback_encoded = [
            game.feedback_to_int[game.CORRECT_POSITION],  # a
            game.feedback_to_int[game.INCORRECT_LETTER],  # u
            game.feedback_to_int[game.INCORRECT_LETTER],  # d
            game.feedback_to_int[game.INCORRECT_LETTER],  # i
            game.feedback_to_int[game.INCORRECT_LETTER],  # o
        ]
        np.testing.assert_array_equal(board1[0, :, 0], expected_guess_chars_encoded)
        np.testing.assert_array_equal(board1[0, :, 1], expected_guess_feedback_encoded)

        # Check remaining turns in board are still padded
        self.assertTrue(np.all(board1[1:, :, 0] == -1))  # Chars are -1
        self.assertTrue(np.all(board1[1:, :, 1] == 0))  # Feedbacks are 0

        # Check alphabet_status
        self.assertEqual(
            alpha_status1[game.char_to_int["a"]],
            game.alphabet_status_to_int[game.CORRECT_POSITION],
        )
        self.assertEqual(
            alpha_status1[game.char_to_int["u"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )
        self.assertEqual(
            alpha_status1[game.char_to_int["d"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )
        self.assertEqual(
            alpha_status1[game.char_to_int["i"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )
        self.assertEqual(
            alpha_status1[game.char_to_int["o"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )
        self.assertEqual(
            alpha_status1[game.char_to_int["p"]],
            game.alphabet_status_to_int[game.STATUS_UNKNOWN],
        )  # p from apple not guessed
        self.assertEqual(
            alpha_status1[game.char_to_int["l"]],
            game.alphabet_status_to_int[game.STATUS_UNKNOWN],
        )  # l from apple not guessed
        self.assertEqual(
            alpha_status1[game.char_to_int["e"]],
            game.alphabet_status_to_int[game.STATUS_UNKNOWN],
        )  # e from apple not guessed

        # Make another guess: "apply" (secret: "apple")
        # Feedback: a:green, p:green, p:green, l:yellow, y:gray
        game.make_guess("apply")
        state_after_guess2 = game.get_state()
        board2 = state_after_guess2["board_representation"]
        alpha_status2 = state_after_guess2["alphabet_status"]
        current_turn2 = state_after_guess2["current_turn"]

        self.assertEqual(current_turn2[0], 2)
        # Check board_representation for the second guess (turn_idx = 1)
        expected_guess2_chars_encoded = [
            game.char_to_int["a"],
            game.char_to_int["p"],
            game.char_to_int["p"],
            game.char_to_int["l"],
            game.char_to_int["y"],
        ]
        expected_guess2_feedback_encoded = [
            game.feedback_to_int[game.CORRECT_POSITION],  # a
            game.feedback_to_int[game.CORRECT_POSITION],  # p
            game.feedback_to_int[game.CORRECT_POSITION],  # p
            game.feedback_to_int[
                game.CORRECT_POSITION
            ],  # l - Corrected from WRONG_POSITION
            game.feedback_to_int[game.INCORRECT_LETTER],  # y
        ]
        np.testing.assert_array_equal(board2[1, :, 0], expected_guess2_chars_encoded)
        np.testing.assert_array_equal(board2[1, :, 1], expected_guess2_feedback_encoded)

        # Check alphabet_status after second guess
        self.assertEqual(
            alpha_status2[game.char_to_int["a"]],
            game.alphabet_status_to_int[game.CORRECT_POSITION],
        )
        self.assertEqual(
            alpha_status2[game.char_to_int["p"]],
            game.alphabet_status_to_int[game.CORRECT_POSITION],
        )
        self.assertEqual(
            alpha_status2[game.char_to_int["l"]],
            game.alphabet_status_to_int[
                game.CORRECT_POSITION
            ],  # Corrected: 'l' is green
        )
        self.assertEqual(
            alpha_status2[game.char_to_int["y"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )
        self.assertEqual(
            alpha_status2[game.char_to_int["e"]],
            game.alphabet_status_to_int[game.STATUS_UNKNOWN],
        )  # e from apple still not revealed by a green/yellow
        # Status of u,d,i,o should remain gray from previous guess
        self.assertEqual(
            alpha_status2[game.char_to_int["u"]],
            game.alphabet_status_to_int[game.INCORRECT_LETTER],
        )

    def test_alphabet_status_update_priority(self):
        game = WordleGame(word_length=5, max_turns=6)
        game.secret_word = "slate"  # s, l, a, t, e

        # Guess 1: "least" -> Feedback: L:yellow, E:yellow, A:green, S:yellow, T:yellow
        # Expected alphabet based on current game behavior: l:yellow, e:unknown, a:green, s:yellow, t:yellow
        game.make_guess("least")
        state1 = game.get_state()["alphabet_status"]
        self.assertEqual(
            state1[game.char_to_int["l"]],
            game.alphabet_status_to_int[
                game.CORRECT_LETTER_WRONG_POSITION
            ],  # l: yellow
        )
        self.assertEqual(
            state1[game.char_to_int["e"]],
            game.alphabet_status_to_int[
                game.CORRECT_LETTER_WRONG_POSITION
            ],  # e: yellow
        )
        self.assertEqual(
            state1[game.char_to_int["a"]],
            game.alphabet_status_to_int[game.CORRECT_POSITION],  # a: green
        )
        self.assertEqual(
            state1[game.char_to_int["s"]],
            game.alphabet_status_to_int[
                game.CORRECT_LETTER_WRONG_POSITION
            ],  # s: yellow
        )
        self.assertEqual(
            state1[game.char_to_int["t"]],
            game.alphabet_status_to_int[
                game.CORRECT_LETTER_WRONG_POSITION
            ],  # t: yellow
        )
