import os
import random
import torch
from typing import List, Optional, Tuple, Callable


class OpponentProvider:
    """Manages opponent sampling strategies for training (Self-Play, Random Agent, Past Checkpoints)."""

    def __init__(
        self,
        opponent_types: List[str],
        opponent_weights: List[float],
        past_checkpoint_dir: Optional[str] = None,
        model_cls: Optional[Callable] = None,
        device: str = "cuda"
    ):
        self.opponent_types = opponent_types
        self.opponent_weights = opponent_weights
        self.past_checkpoint_dir = past_checkpoint_dir
        self.model_cls = model_cls
        self.device = device
        self._past_checkpoints: List[str] = []

        if past_checkpoint_dir and os.path.exists(past_checkpoint_dir):
            self._scan_past_checkpoints()

    def _scan_past_checkpoints(self):
        if self.past_checkpoint_dir and os.path.exists(self.past_checkpoint_dir):
            self._past_checkpoints = [
                os.path.join(self.past_checkpoint_dir, f)
                for f in os.listdir(self.past_checkpoint_dir)
                if f.endswith(".pt") or f.endswith(".pth")
            ]

    def sample_opponent_type(self) -> str:
        """Samples opponent type based on configured weights."""
        if not self.opponent_types:
            return "self_play"
        return random.choices(self.opponent_types, weights=self.opponent_weights, k=1)[0]

    def get_opponent_model(self, opponent_type: str, current_model: torch.nn.Module) -> Tuple[str, Optional[torch.nn.Module]]:
        """Returns the model instance to be used for P2/Opponent in a match."""
        if opponent_type == "self_play":
            # Uses current model weights for self-play
            return "self_play", current_model
        
        elif opponent_type == "random":
            # Pure random agent (no neural net required)
            return "random", None

        elif opponent_type == "past_checkpoint":
            self._scan_past_checkpoints()
            if self._past_checkpoints and self.model_cls is not None:
                ckpt_path = random.choice(self._past_checkpoints)
                try:
                    opp_model = self.model_cls().to(self.device)
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    opp_model.load_state_dict(state_dict)
                    opp_model.eval()
                    return f"past_checkpoint:{os.path.basename(ckpt_path)}", opp_model
                except Exception as e:
                    print(f"Warning: Failed to load opponent checkpoint {ckpt_path}: {e}")
            
            # Fallback to self_play if checkpoint loading fails
            return "self_play", current_model

        return "self_play", current_model
