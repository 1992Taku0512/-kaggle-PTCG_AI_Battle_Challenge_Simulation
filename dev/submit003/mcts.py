import math
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from state_encoder import StateEncoder
from model import AlphaZeroNet


class MCTSNode:
    """Represents a node in the MCTS tree"""
    def __init__(self, action_idx: Optional[int] = None, prior: float = 0.0):
        self.action_idx = action_idx
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class AlphaZeroMCTS:
    """AlphaZero MCTS Search Engine for PTCG Agent

    Combines StateEncoder, AlphaZeroNet (GPU), and PUCT search to determine
    the optimal option choice for a given Observation.
    """
    def __init__(
        self,
        model: AlphaZeroNet,
        encoder: StateEncoder,
        c_puct: float = 1.414,
        num_simulations: int = 50,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def get_action_distribution(self, obs: Any) -> Tuple[List[int], np.ndarray]:
        """Runs MCTS search and returns option indices and visit count probability distribution"""
        select_info = getattr(obs, "select", None)
        if select_info is None:
            return [0], np.array([1.0], dtype=np.float32)

        options = getattr(select_info, "option", [])
        num_options = len(options)
        if num_options == 0:
            return [0], np.array([1.0], dtype=np.float32)

        # 1. Encode observation state to Tensor on GPU
        state_tensor = self.encoder.encode(obs).to(self.device).unsqueeze(0)

        # 2. Evaluate with Neural Network (Policy & Value)
        with torch.no_grad():
            policy_logits, state_value = self.model(state_tensor)
            policy_probs = torch.softmax(policy_logits[0], dim=-1).cpu().numpy()
            value = state_value[0, 0].item()

        # 3. Create Root Node and populate children for each valid option index
        root = MCTSNode()
        for opt_idx in range(num_options):
            # Map option index to prior probability
            prior_p = float(policy_probs[opt_idx % len(policy_probs)])
            root.children[opt_idx] = MCTSNode(action_idx=opt_idx, prior=prior_p)

        # 4. MCTS Simulations using PUCT formula
        for _ in range(self.num_simulations):
            total_visits = sum(child.visit_count for child in root.children.values())
            best_opt_idx = None
            best_puct = -float("inf")

            for opt_idx, child in root.children.items():
                u_score = self.c_puct * child.prior * (math.sqrt(total_visits + 1e-8) / (1 + child.visit_count))
                puct_score = child.q_value + u_score
                if puct_score > best_puct:
                    best_puct = puct_score
                    best_opt_idx = opt_idx

            if best_opt_idx is not None:
                # Virtual rollout step update
                chosen_child = root.children[best_opt_idx]
                chosen_child.visit_count += 1
                chosen_child.total_value += value

        # 5. Calculate probabilities based on visit counts
        visits = np.array([root.children[i].visit_count for i in range(num_options)], dtype=np.float32)
        total_v = visits.sum()
        probs = visits / total_v if total_v > 0 else np.ones(num_options, dtype=np.float32) / num_options

        best_option_idx = int(np.argmax(probs))
        return [best_option_idx], probs


if __name__ == "__main__":
    encoder = StateEncoder()
    model = AlphaZeroNet()
    mcts = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=30)
    print(f"✅ AlphaZeroMCTS initialized successfully on device: {mcts.device}")
