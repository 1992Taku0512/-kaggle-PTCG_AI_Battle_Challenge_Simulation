import math
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from state_encoder import StateEncoder
from model import AlphaZeroNet


def extract_option_type_val(opt: Any) -> int:
    """Helper to extract OptionType integer regardless of whether opt is a dict or dataclass."""
    if isinstance(opt, dict):
        raw_type = opt.get("type", 0)
    else:
        raw_type = getattr(opt, "type", 0)

    if hasattr(raw_type, "value"):
        return int(raw_type.value)
    try:
        return int(raw_type)
    except (ValueError, TypeError):
        return 0


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

    def is_leaf(self) -> float:
        return len(self.children) == 0


class AlphaZeroMCTS:
    """AlphaZero MCTS Search Engine for PTCG Agent with Domain-Heuristic Policy Priors"""
    def __init__(
        self,
        model: AlphaZeroNet,
        encoder: StateEncoder,
        c_puct: float = 1.5,
        num_simulations: int = 40,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _get_option_logits_idx(self, opt: Any) -> int:
        """Map option type/parameters to a policy logit index (0 to 63)"""
        type_val = extract_option_type_val(opt)
        return min(type_val, 63)

    def _get_action_heuristic_bonus(self, opt: Any, has_active_options: bool) -> float:
        """Applies heuristic bonus/penalty to prevent premature turn ending."""
        type_val = extract_option_type_val(opt)

        # 13: ATTACK -> Highest Priority
        if type_val == 13:
            return 8.0
        # 8: ATTACH (Energy) -> High Priority
        elif type_val == 8:
            return 5.0
        # 7: PLAY (Item / Supporter / Basic Pokemon) / 9: EVOLVE / 10: ABILITY
        elif type_val in (7, 9, 10):
            return 3.0
        # 14: END (Turn End / Pass) -> Penalty if other productive actions exist
        elif type_val == 14 and has_active_options:
            return -8.0
        return 0.0

    def get_action_distribution(self, obs: Any) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Runs MCTS search and returns:
        - Chosen option index list
        - Option visit count probability distribution
        - Full 64-dim Policy Target vector for training Policy Head
        """
        if isinstance(obs, dict):
            select_info = obs.get("select")
        else:
            select_info = getattr(obs, "select", None)

        target_policy_vector = np.zeros(64, dtype=np.float32)

        if select_info is None:
            target_policy_vector[0] = 1.0
            return [0], np.array([1.0], dtype=np.float32), target_policy_vector

        if isinstance(select_info, dict):
            options = select_info.get("option", [])
        else:
            options = getattr(select_info, "option", [])

        num_options = len(options)
        if num_options == 0:
            target_policy_vector[0] = 1.0
            return [0], np.array([1.0], dtype=np.float32), target_policy_vector

        # Check if there are active actions besides END TURN (14)
        has_active = any(
            extract_option_type_val(opt) in (7, 8, 9, 10, 13)
            for opt in options
        )

        # 1. Encode observation state to Tensor on GPU
        state_tensor = self.encoder.encode(obs).to(self.device).unsqueeze(0)

        # 2. Evaluate with Neural Network (Policy & Value)
        with torch.no_grad():
            policy_logits, state_value = self.model(state_tensor)
            logits_np = policy_logits[0].cpu().numpy()
            value = state_value[0, 0].item()

        # 3. Extract Prior Logits with Heuristic Guidance
        option_logits = []
        for opt in options:
            logit_idx = self._get_option_logits_idx(opt)
            base_logit = logits_np[logit_idx]
            bonus = self._get_action_heuristic_bonus(opt, has_active_options=has_active)
            option_logits.append(base_logit + bonus)

        option_logits = np.array(option_logits, dtype=np.float32)

        # Numerical stability for Softmax
        exp_logits = np.exp(option_logits - np.max(option_logits))
        priors = exp_logits / (np.sum(exp_logits) + 1e-8)

        # 4. Create Root Node with Policy Head Priors
        root = MCTSNode()
        for opt_idx in range(num_options):
            root.children[opt_idx] = MCTSNode(action_idx=opt_idx, prior=float(priors[opt_idx]))

        # 5. MCTS Simulations using PUCT formula
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
                chosen_child = root.children[best_opt_idx]
                chosen_child.visit_count += 1
                chosen_child.total_value += value

        # 6. Calculate visit count probability distribution
        visits = np.array([root.children[i].visit_count for i in range(num_options)], dtype=np.float32)
        total_v = visits.sum()
        probs = visits / total_v if total_v > 0 else np.ones(num_options, dtype=np.float32) / num_options

        # Populate target policy vector (64-dim) for GPU Policy Head Loss Training
        for opt_idx, opt in enumerate(options):
            logit_idx = self._get_option_logits_idx(opt)
            target_policy_vector[logit_idx] += probs[opt_idx]

        target_policy_sum = target_policy_vector.sum()
        if target_policy_sum > 0:
            target_policy_vector /= target_policy_sum
        else:
            target_policy_vector[0] = 1.0

        best_option_idx = int(np.argmax(probs))
        return [best_option_idx], probs, target_policy_vector


if __name__ == "__main__":
    encoder = StateEncoder()
    model = AlphaZeroNet()
    mcts = AlphaZeroMCTS(model=model, encoder=encoder, num_simulations=30)
    print(f"✅ AlphaZeroMCTS Policy Prior engine with robust dict/dataclass support initialized on device: {mcts.device}")
