import math
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

if "__file__" in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = "/kaggle_simulations/agent" if os.path.exists("/kaggle_simulations/agent") else os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if os.path.exists(sample_dir) and sample_dir not in sys.path:
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
    """Represents a node in the Prioritized MCTS tree."""

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
    """Prioritized AlphaZero MCTS Engine supporting dynamic min_p_threshold curriculum."""

    def __init__(
        self,
        model: TransformerAlphaZeroNet,
        encoder: StateEncoder,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        min_p_threshold: float = 0.0,   # Default 0.0 (No pruning in Phase 1)
        max_pruned_candidates: int = 8,
        min_candidates: int = 4          # Top-4 guarantee
    ):
        self.model = model
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.min_p_threshold = min_p_threshold
        self.max_pruned_candidates = max_pruned_candidates
        self.min_candidates = min_candidates
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

        exp_priors = np.exp(priors - np.max(priors))
        policy_probs = exp_priors / (np.sum(exp_priors) + 1e-8)

        return value, policy_probs

    def _filter_candidate_actions(self, priors: np.ndarray, thresh: Optional[float] = None) -> List[int]:
        """Prioritized Action Pruning with Top-4 Guarantee and dynamic threshold."""
        if thresh is None:
            thresh = self.min_p_threshold

        num_options = len(priors)
        if num_options <= self.min_candidates or thresh <= 0.0:
            return list(range(num_options))

        sorted_indices = np.argsort(priors)[::-1]
        candidates = []
        cum_prob = 0.0

        for idx in sorted_indices:
            candidates.append(int(idx))
            cum_prob += priors[idx]
            if len(candidates) >= self.min_candidates:
                if len(candidates) >= self.max_pruned_candidates or cum_prob >= 0.95 or priors[idx] < thresh:
                    break

        return candidates

    def get_action_distribution(
        self,
        obs: Observation,
        your_deck: List[int] = None,
        opp_deck: List[int] = None,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,
        min_p_threshold: Optional[float] = None
    ) -> Tuple[List[int], int, np.ndarray]:
        """Runs MCTS search with Search API integration."""
        if your_deck is None:
            your_deck = [3] * 60
        if opp_deck is None:
            opp_deck = [3] * 60

        if not obs.select or not obs.select.option:
            return [0], 0, np.array([1.0])

        num_options = len(obs.select.option)
        val, priors = self.evaluate_or_default(obs, your_deck, num_options)

        if dirichlet_epsilon > 0.0 and len(priors) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(priors))
            priors = (1.0 - dirichlet_epsilon) * priors + dirichlet_epsilon * noise

        root = MCTSNode()
        candidate_indices = self._filter_candidate_actions(priors, min_p_threshold)
        for idx in candidate_indices:
            root.children[idx] = MCTSNode(action_idx=idx, prior=priors[idx])

        for _ in range(self.num_simulations):
            try:
                search_state = search_begin(your_deck, opp_deck)
            except Exception:
                break

            node = root
            search_obs = obs
            sim_done = False

            while not node.is_leaf() and not sim_done:
                total_visits = sum(child.visit_count for child in node.children.values())
                best_score = -float('inf')
                best_action = candidate_indices[0] if candidate_indices else 0

                for act_idx, child in node.children.items():
                    u = self.c_puct * child.prior * (math.sqrt(total_visits + 1e-8) / (1 + child.visit_count))
                    score = child.q_value + u
                    if score > best_score:
                        best_score = score
                        best_action = act_idx

                try:
                    res_obs = search_step(search_state, [best_action])
                    if res_obs is None:
                        sim_done = True
                    else:
                        search_obs = to_observation_class(res_obs)
                except Exception:
                    sim_done = True

                node = node.children[best_action]

            if not sim_done and search_obs and search_obs.select and search_obs.select.option:
                leaf_val, leaf_priors = self.evaluate_or_default(search_obs, your_deck, len(search_obs.select.option))
                leaf_candidates = self._filter_candidate_actions(leaf_priors, min_p_threshold)
                for idx in leaf_candidates:
                    node.children[idx] = MCTSNode(action_idx=idx, prior=leaf_priors[idx])
            else:
                leaf_val = val

            node.visit_count += 1
            node.total_value += leaf_val

            try:
                search_end(search_state)
            except Exception:
                pass

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
