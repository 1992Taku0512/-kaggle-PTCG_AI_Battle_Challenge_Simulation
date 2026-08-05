import math
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.api import (
    Observation,
    OptionType,
    search_begin,
    search_step,
    search_end,
    to_observation_class,
)
from state_encoder import StateEncoder, SparseVector
from model import TransformerAlphaZeroNet


class MCTSNode:
    """Represents a node in the MCTS tree."""

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
    """AlphaZero MCTS Search Engine using Official Search API (search_begin, search_step, search_end)."""

    def __init__(
        self,
        model: TransformerAlphaZeroNet,
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

    def _evaluate_state(self, obs: Observation, your_deck: List[int]) -> Tuple[float, np.ndarray]:
        """Evaluates board state using Transformer NN."""
        sv_enc = self.encoder.encode(obs, your_deck)
        sv_dec = self.encoder.encode_decoder(obs, your_deck)

        if not sv_dec.offset:
            return 0.0, np.array([])

        idx_enc = torch.tensor(sv_enc.index, dtype=torch.int64, device=self.device)
        val_enc = torch.tensor(sv_enc.value, dtype=torch.float32, device=self.device)
        off_enc = torch.tensor(sv_enc.offset, dtype=torch.int64, device=self.device)

        idx_dec = torch.tensor(sv_dec.index, dtype=torch.int64, device=self.device)
        val_dec = torch.tensor(sv_dec.value, dtype=torch.float32, device=self.device)
        off_dec = torch.tensor(sv_dec.offset, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            value_tensor, policy_tensor = self.model(
                idx_enc, val_enc, off_enc,
                idx_dec, val_dec, off_dec
            )

        value = value_tensor.item()
        priors = policy_tensor.squeeze(0).cpu().numpy()

        # Softmax normalization over options
        exp_priors = np.exp(priors - np.max(priors))
        policy_probs = exp_priors / (np.sum(exp_priors) + 1e-8)

        return value, policy_probs

    def get_action_distribution(self, obs: Observation, your_deck: List[int] = None, opp_deck: List[int] = None) -> Tuple[List[int], int, np.ndarray]:
        """Runs MCTS search with Search API integration and returns action distribution."""
        if your_deck is None:
            your_deck = [1002] * 60
        if opp_deck is None:
            opp_deck = [1002] * 60

        if not obs.select or not obs.select.option:
            return [0], 0, np.array([1.0])

        num_options = len(obs.select.option)
        val, priors = self.evaluate_or_default(obs, your_deck, num_options)

        root = MCTSNode()
        for idx in range(num_options):
            root.children[idx] = MCTSNode(action_idx=idx, prior=priors[idx])

        # Execute MCTS simulations via Search API
        for _ in range(self.num_simulations):
            try:
                # 1. Search API Begin (Determinization)
                search_state = search_begin(your_deck, opp_deck)
            except Exception:
                break

            node = root
            search_obs = obs
            sim_done = False

            # Traverse tree
            while not node.is_leaf() and not sim_done:
                # PUCT Selection
                total_visits = sum(child.visit_count for child in node.children.values())
                best_score = -float('inf')
                best_action = 0

                for act_idx, child in node.children.items():
                    u = self.c_puct * child.prior * (math.sqrt(total_visits) / (1 + child.visit_count))
                    score = child.q_value + u
                    if score > best_score:
                        best_score = score
                        best_action = act_idx

                # 2. Search API Step
                try:
                    res_obs = search_step(search_state, [best_action])
                    if res_obs is None:
                        sim_done = True
                    else:
                        search_obs = to_observation_class(res_obs)
                except Exception:
                    sim_done = True

                node = node.children[best_action]

            # Evaluate leaf & backpropagate
            if not sim_done and search_obs and search_obs.select and search_obs.select.option:
                leaf_val, leaf_priors = self.evaluate_or_default(search_obs, your_deck, len(search_obs.select.option))
                for idx, p_val in enumerate(leaf_priors):
                    node.children[idx] = MCTSNode(action_idx=idx, prior=p_val)
            else:
                leaf_val = val

            # Backprop
            node.visit_count += 1
            node.total_value += leaf_val

            # 3. Search API End
            try:
                search_end(search_state)
            except Exception:
                pass

        # Calculate final visits & policy target
        visits = np.array([root.children[i].visit_count if i in root.children else 0 for i in range(num_options)])
        total_visits = np.sum(visits)
        if total_visits > 0:
            policy_target = visits / total_visits
            best_opt_idx = int(np.argmax(visits))
        else:
            policy_target = np.ones(num_options) / num_options
            best_opt_idx = 0

        return [best_opt_idx], best_opt_idx, policy_target

    def evaluate_or_default(self, obs: Observation, your_deck: List[int], num_options: int) -> Tuple[float, np.ndarray]:
        try:
            val, priors = self._evaluate_state(obs, your_deck)
            if len(priors) != num_options:
                priors = np.ones(num_options) / num_options
            return val, priors
        except Exception:
            return 0.0, np.ones(num_options) / num_options
