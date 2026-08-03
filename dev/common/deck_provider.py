import os
import random
from typing import List, Tuple


class DeckProvider:
    """Manages loading and sampling of decks for Player 1 and Player 2."""

    def __init__(self, project_root: str, candidate_paths: List[str] = None):
        self.project_root = project_root
        self.decks: List[Tuple[str, List[int]]] = []
        
        if candidate_paths:
            for path in candidate_paths:
                abs_path = path if os.path.isabs(path) else os.path.join(project_root, path)
                if os.path.exists(abs_path):
                    self._add_deck_file(abs_path)
        
        # If no decks loaded or candidate_paths was empty, load standard deck pool from dev/deck_pool
        if not self.decks:
            deck_pool_dir = os.path.join(project_root, "dev", "deck_pool")
            if os.path.exists(deck_pool_dir):
                for fname in os.listdir(deck_pool_dir):
                    if fname.endswith(".csv"):
                        self._add_deck_file(os.path.join(deck_pool_dir, fname))

        # Fallback to sample submission deck if still empty
        if not self.decks:
            sample_deck_path = os.path.join(
                project_root, "data", "sample_submission", "sample_submission", "deck.csv"
            )
            if os.path.exists(sample_deck_path):
                self._add_deck_file(sample_deck_path)

    def _add_deck_file(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            deck_ids = [int(line.strip()) for line in lines if line.strip() and not line.startswith("#")][:60]
            if len(deck_ids) == 60:
                basename = os.path.basename(filepath)
                self.decks.append((basename, deck_ids))
        except Exception as e:
            print(f"Warning: Failed to load deck file {filepath}: {e}")

    def sample_deck_pair(self, p1_mode: str = "random", p2_mode: str = "random") -> Tuple[Tuple[str, List[int]], Tuple[str, List[int]]]:
        """Samples a pair of decks for P1 and P2 based on sampling mode."""
        if not self.decks:
            raise RuntimeError("No valid 60-card decks found in DeckProvider.")

        p1_deck = self.decks[0] if p1_mode == "fixed" else random.choice(self.decks)
        p2_deck = self.decks[0] if p2_mode == "fixed" else random.choice(self.decks)

        return p1_deck, p2_deck
