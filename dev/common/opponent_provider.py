import os
import sys
import random
import torch
from typing import List, Optional, Tuple, Callable, Any

# Load sampleAgent001, sampleAgent002, submit009 agent functions
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

def load_external_agent(agent_dir: str):
    full_path = os.path.join(project_root, agent_dir)
    if os.path.exists(full_path) and full_path not in sys.path:
        sys.path.insert(0, full_path)
    try:
        mod_name = f"dev_{os.path.basename(agent_dir)}_main"
        import importlib.util
        main_py = os.path.join(full_path, "main.py")
        if os.path.exists(main_py):
            spec = importlib.util.spec_from_file_location(mod_name, main_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "agent", None)
    except Exception as e:
        print(f"Warning: Failed to load agent from {agent_dir}: {e}")
    return None

agent_sample001 = load_external_agent("dev/sampleAgent001")
agent_sample002 = load_external_agent("dev/sampleAgent002")
agent_submit009 = load_external_agent("dev/submit009")


class OpponentProvider:
    """Manages opponent sampling strategies for training (Self-Play, sampleAgent002, submit009, sampleAgent001, etc)."""

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

    def get_opponent_model(self, opponent_type: str, current_model: torch.nn.Module) -> Tuple[str, Optional[Any]]:
        """Returns the model or agent function instance to be used for P2/Opponent in a match."""
        if opponent_type == "self_play":
            return "self_play", current_model

        elif opponent_type == "sampleAgent002":
            if agent_sample002 is not None:
                return "sampleAgent002", agent_sample002
            return "random", None

        elif opponent_type == "submit009":
            if agent_submit009 is not None:
                return "submit009", agent_submit009
            return "random", None

        elif opponent_type == "sampleAgent001":
            if agent_sample001 is not None:
                return "sampleAgent001", agent_sample001
            return "random", None

        elif opponent_type in ("random", "official_sample"):
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
                    return f"past_ckpt({os.path.basename(ckpt_path)})", opp_model
                except Exception as e:
                    print(f"Warning: Failed to load opponent checkpoint {ckpt_path}: {e}")

        return "self_play", current_model
