import os
import sys
import time
import re
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.game import battle_start, battle_select
from cg.api import to_observation_class, OptionType
from dev.common.config import TrainerConfig
from dev.common.deck_provider import DeckProvider
from dev.common.opponent_provider import OpponentProvider

try:
    from notify_line import send_line_notification
except ImportError:
    def send_line_notification(msg: str):
        pass


def strip_ansi(text: str) -> str:
    """Strips ANSI control/escape characters from string."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def extract_active_player(obs_dict: dict) -> int:
    """Extract active player index reliably from obs_dict."""
    if isinstance(obs_dict, dict):
        if "player" in obs_dict:
            return obs_dict["player"]
        if "current" in obs_dict and isinstance(obs_dict["current"], dict):
            return obs_dict["current"].get("yourIndex", 0)
    return 0


def check_match_finish(obs_dict: dict) -> Tuple[bool, int]:
    """Check match completion and winner player index."""
    if isinstance(obs_dict, dict):
        if "winner" in obs_dict and obs_dict["winner"] is not None:
            return True, obs_dict["winner"]
        if "result" in obs_dict and isinstance(obs_dict["result"], dict):
            res = obs_dict["result"]
            if "winner" in res and res["winner"] is not None:
                return True, res["winner"]
    return False, -1


class PTCGTrainer:
    """Generic Model-Agnostic Reinforcement Learning Trainer & Wrapper for PTCG AI."""

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        encoder: Any,
        mcts_cls: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_cls: Optional[Callable] = None
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.encoder = encoder
        self.mcts_cls = mcts_cls
        self.model_cls = model_cls or (lambda: model.__class__())

        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)
        
        self.deck_provider = DeckProvider(project_root, candidate_paths=config.deck_pool_paths)
        self.opponent_provider = OpponentProvider(
            opponent_types=config.opponent_types,
            opponent_weights=config.opponent_weights,
            past_checkpoint_dir=config.past_checkpoint_dir,
            model_cls=self.model_cls,
            device=config.device
        )
        
        self.experience_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 30000
        
        self.start_episode = 1
        self.best_winrate = 0.0
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Resume checkpoint if specified
        if config.resume_checkpoint_path:
            self._load_checkpoint(config.resume_checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        abs_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(project_root, checkpoint_path)
        if not os.path.exists(abs_path):
            print(f"Warning: Checkpoint path '{abs_path}' not found. Initializing brand-new model weights.")
            return

        print(f"Loading checkpoint from: {abs_path}")
        ckpt = torch.load(abs_path, map_location=self.device)
        
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "episode" in ckpt:
                self.start_episode = ckpt["episode"] + 1
            if "best_winrate" in ckpt:
                self.best_winrate = ckpt["best_winrate"]
            print(f"Resumed successfully from Episode {self.start_episode - 1}")
        else:
            self.model.load_state_dict(ckpt)
            print("Resumed model weights directly.")

    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Saves checkpoint to disk."""
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"model_ep{episode}.pt")
        state = {
            "episode": episode,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_winrate": self.best_winrate,
            "config": self.config
        }
        torch.save(state, ckpt_path)
        
        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        latest_path = os.path.join(self.config.checkpoint_dir, "latest_model.pt")
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, best_path)
            print(f"🏆 New best model saved to {best_path} (Winrate: {self.best_winrate:.1f}%)")

    def run_eval_subprocess(self, agent1_dir: str, agent2_dir: str = "data/sample_submission/sample_submission") -> Dict[str, Any]:
        """Runs eval_local.py in an isolated subprocess."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["COLUMNS"] = "200"
        
        cmd = [
            sys.executable,
            "eval_local.py",
            "--agent1", agent1_dir,
            "--agent2", agent2_dir,
            "--num-games", str(self.config.eval_num_games)
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=env)
        clean_stdout = strip_ansi(res.stdout)
        
        a1_wins = 0
        winrate = 0.0
        avg_turns = 0.0

        match_wins = re.search(r"Agent 1.*?Wins\s+:\s+(\d+)\s+/\s+(\d+)\s+\(([\d.]+)%\)", clean_stdout, re.DOTALL)
        if match_wins:
            a1_wins = int(match_wins.group(1))
            winrate = float(match_wins.group(3))

        match_turns = re.search(r"Average Turn Count\s+:\s+([\d.]+)", clean_stdout)
        if match_turns:
            avg_turns = float(match_turns.group(1))

        return {
            "a1_wins": a1_wins,
            "winrate": winrate,
            "avg_turns": avg_turns,
            "raw_output": clean_stdout
        }

    def collect_self_play_episode(self, d0: List[int], d1: List[int], opp_model: Optional[nn.Module] = None) -> List[Dict[str, Any]]:
        """Simulates one match and computes TD(lambda) smoothed value targets."""
        if self.mcts_cls is None:
            return []

        # Instantiates MCTS engines
        p1_mcts = self.mcts_cls(model=self.model, encoder=self.encoder, num_simulations=self.config.search_count, device=self.device)
        p2_mcts = self.mcts_cls(model=opp_model or self.model, encoder=self.encoder, num_simulations=self.config.search_count, device=self.device)

        obs_dict, _ = battle_start(d0, d1)
        turn_count = 0
        max_turns = 150
        episode_experiences = []
        winner = -1

        while obs_dict is not None and turn_count < max_turns:
            turn_count += 1
            is_done, match_winner = check_match_finish(obs_dict)
            if is_done:
                winner = match_winner
                break

            current_player = extract_active_player(obs_dict)
            obs = to_observation_class(obs_dict)

            if obs.select is None:
                action = d0 if current_player == 0 else d1
            else:
                state_vec = self.encoder.encode(obs)
                active_mcts = p1_mcts if current_player == 0 else p2_mcts
                
                action_list, _, policy_target = active_mcts.get_action_distribution(obs)
                action = action_list

                episode_experiences.append({
                    "state": state_vec,
                    "policy_target": policy_target,
                    "player": current_player,
                })

            try:
                obs_dict = battle_select(action)
            except Exception:
                break

        if obs_dict is not None and winner == -1:
            is_done, match_winner = check_match_finish(obs_dict)
            if is_done:
                winner = match_winner

        if not episode_experiences:
            return []

        # TD(lambda) backward smoothing for value targets
        final_value_p0 = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)
        running_value_p0 = final_value_p0
        running_value_p1 = -final_value_p0

        for exp in reversed(episode_experiences):
            player = exp["player"]
            if player == 0:
                target_val = running_value_p0
                running_value_p0 = running_value_p0 * self.config.td_lambda + target_val * (1.0 - self.config.td_lambda)
            else:
                target_val = running_value_p1
                running_value_p1 = running_value_p1 * self.config.td_lambda + target_val * (1.0 - self.config.td_lambda)
            exp["value_target"] = target_val

        return episode_experiences

    def update_model(self) -> float:
        """Performs mini-batch SGD update on the PyTorch neural network."""
        if len(self.experience_buffer) < self.config.batch_size:
            return 0.0

        batch_samples = random.sample(self.experience_buffer, self.config.batch_size)
        
        states = torch.tensor(np.array([s["state"] for s in batch_samples]), dtype=torch.float32, device=self.device)
        policy_targets = torch.tensor(np.array([s["policy_target"] for s in batch_samples]), dtype=torch.float32, device=self.device)
        value_targets = torch.tensor([[s["value_target"]] for s in batch_samples], dtype=torch.float32, device=self.device)

        self.model.train()
        pred_policy, pred_value = self.model(states)

        policy_loss = F.huber_loss(pred_policy, policy_targets, delta=0.1)
        value_loss = F.huber_loss(pred_value, value_targets, delta=0.2)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(self, agent_save_dir: Optional[str] = None):
        """Main training, evaluation, and notification execution loop."""
        print(f"================================================================================")
        print(f"🚀 Starting PTCG Training: {self.config.experiment_name}")
        print(f"   Device: {self.device} | Range: Ep {self.start_episode} -> Ep {self.config.num_episodes}")
        print(f"   Checkpoints Path: {self.config.checkpoint_dir}")
        print(f"================================================================================")
        
        start_time = time.time()
        last_loss = 0.0

        for ep in range(self.start_episode, self.config.num_episodes + 1):
            # 1. Sample decks and opponent dynamically
            (p1_name, p1_deck), (p2_name, p2_deck) = self.deck_provider.sample_deck_pair(
                self.config.p1_deck_mode, self.config.p2_deck_mode
            )
            opp_type = self.opponent_provider.sample_opponent_type()
            opp_desc, opp_model = self.opponent_provider.get_opponent_model(opp_type, self.model)

            # 2. Collect episode experiences
            ep_samples = self.collect_self_play_episode(p1_deck, p2_deck, opp_model=opp_model)
            self.experience_buffer.extend(ep_samples)
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

            # 3. Train batch update
            if len(self.experience_buffer) >= self.config.batch_size:
                last_loss = self.update_model()

            # Logging info
            if ep % 50 == 0 or ep == 1:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"[Ep {ep:4d}/{self.config.num_episodes}] Elapsed: {elapsed} | Loss: {last_loss:.4f} | Decks: P1({p1_name}) vs P2({p2_name}) | Opp: {opp_desc}")

            # 4. Checkpoint & Evaluation
            if ep % self.config.eval_every == 0 and agent_save_dir:
                print(f"\n📊 Running Evaluation Benchmark at Episode {ep}...")
                
                # Save latest model weights to agent directory for evaluation
                if hasattr(self.model, "state_dict"):
                    weights_dst = os.path.join(project_root, agent_save_dir, "model_weights.pt")
                    torch.save(self.model.state_dict(), weights_dst)

                eval_res = self.run_eval_subprocess(agent_save_dir)
                wr = eval_res["winrate"]
                turns = eval_res["avg_turns"]
                print(f"-> Eval Result vs Sample: Winrate {wr:.1f}% ({eval_res['a1_wins']}/{self.config.eval_num_games}) | Avg Turns: {turns:.1f}\n")

                is_best = wr > self.best_winrate
                if is_best:
                    self.best_winrate = wr

                self.save_checkpoint(ep, is_best=is_best)

                # LINE Notification
                if self.config.use_line_notify and (ep % self.config.line_notify_every == 0 or is_best):
                    msg = (
                        f"🏆 [{self.config.experiment_name}]\n"
                        f"Episode: {ep}/{self.config.num_episodes}\n"
                        f"Winrate: {wr:.1f}% ({'NEW BEST!' if is_best else f'Best: {self.best_winrate:.1f}%'})\n"
                        f"Avg Turns: {turns:.1f} | Loss: {last_loss:.4f}"
                    )
                    send_line_notification(msg)

        print(f"\n🎉 Training Successfully Completed for {self.config.experiment_name}!")
