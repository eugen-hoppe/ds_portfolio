import re

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Corp:
    """
    Class holding the vocabulary, pair statistics, and BPE ranks.

    Attributes:
        end_of_word: Special marker to indicate the end of a word.
        inf: A representation of infinity to compare against unknown pair ranks.
        vocabulary: A dict mapping tokenized words to their frequencies.
        pairs: A dict mapping symbol pairs (tuples) to their frequencies.
        merged: Temporary dict for merged vocabulary between BPE iterations.
        bpe_ranks: A dict mapping symbol pairs to their merge rank.
    """

    end_of_word: str = "_"
    inf: float = float("inf")
    vocabulary: dict[str, int] = field(default_factory=dict)
    pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    merged: dict[str, int] = field(default_factory=dict)
    bpe_ranks: dict[tuple[str, str], int] = field(default_factory=dict)

    def build_vocabulary(self, corpus: str, separate_punctuation: bool = False) -> None:
        """
        Tokenize the input corpus into words, and build an initial vocabulary
        where each word is split into individual characters plus an end-of-word
        marker.

        Args:
            corpus: A string representing the text corpus.
            separate_punctuation:
              - If True, separate punctuation from words (basic approach).
        """
        # Optionally handle punctuation by inserting spaces around it
        if separate_punctuation:
            corpus = re.sub(r"([.,!?])", r" \1 ", corpus)

        word_counts = Counter(corpus.split())
        for word, count in word_counts.items():
            # Split the word into characters, then add the end_of_word marker
            tokenized_word = " ".join(list(word)) + " " + self.end_of_word
            self.vocabulary[tokenized_word] = count

    def statistics(self) -> None:
        """
        Compute pair frequencies for the current vocabulary. This resets
        and rebuilds the 'pairs' dictionary each time it is called.
        """
        self.pairs.clear()
        for word, freq in self.vocabulary.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):  # Count each adjacent pair
                pair = (symbols[i], symbols[i + 1])
                self.pairs[pair] = self.pairs.get(pair, 0) + freq

    def merge(self, pair: tuple[str, str]) -> None:
        """
        Merge all occurrences of the given pair in the vocabulary.

        Args:
            pair: A tuple (symbol1, symbol2) to be merged into symbol1+symbol2.
        """
        # Build a new vocabulary with the merged token
        pair_str = " ".join(pair)
        pair_merged = "".join(pair)
        self.merged.clear()
        for word, freq in self.vocabulary.items():
            # Replace the space-separated pair with the single merged token
            merged_word = word.replace(pair_str, pair_merged)
            self.merged[merged_word] = freq

    def fit(self, num_merges: int) -> None:
        """
        Perform the BPE merge operations up to 'num_merges' times or until no
        more pairs exist.
        The result is stored in self.vocabulary and self.bpe_ranks.

        Args:
            num_merges: The maximum number of merges to perform.
        """
        for i in range(num_merges):
            # Compute pair frequencies
            self.statistics()
            if not self.pairs:
                break

            # Exclude pairs that contain the end_of_word marker
            valid_pairs = {
                p: freq for p, freq in self.pairs.items() if self.end_of_word not in p
            }

            if not valid_pairs:
                break

            # Find the pair with the highest frequency
            best_pair = max(valid_pairs, key=valid_pairs.get)
            # Assign a rank to this pair (lower rank = earlier merge)
            self.bpe_ranks[best_pair] = i

            # Merge it throughout the vocabulary
            self.merge(best_pair)
            # Update vocabulary from the merged dictionary
            self.vocabulary = self.merged.copy()

    @staticmethod
    def get_pairs_from_tokens(tokens: list[str]) -> set[tuple[str, str]]:
        """
        Given a list of tokens, return a set of adjacent pairs.

        Args:
            tokens: A list of string tokens.

        Returns:
            A set of tuples representing adjacent pairs in 'tokens'.
        """
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def bpe_encode(word: str, corp: Corp) -> list[str]:
    """
    Encode a single word using the BPE merge operations (corp.bpe_ranks).

    Args:
        word: The input word to encode.
        corp: An instance of Corp containing learned BPE ranks and settings.

    Returns:
        A list of merged tokens after applying BPE merges in the order
        specified by corp.bpe_ranks.
    """
    # Start from the basic character-level tokens
    tokens = list(word) + [corp.end_of_word]

    while True:
        # Get all possible adjacent pairs in the tokens
        pairs = Corp.get_pairs_from_tokens(tokens)
        if not pairs:
            break

        # Among all adjacent pairs, choose the one with the lowest rank
        best_pair = min(pairs, key=lambda p: corp.bpe_ranks.get(p, corp.inf))

        # If this pair isn't in bpe_ranks, we won't merge it
        if corp.bpe_ranks.get(best_pair, corp.inf) == corp.inf:
            break

        # Merge occurrences of best_pair
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                # Merge the pair
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        tokens = merged_tokens

    return tokens
