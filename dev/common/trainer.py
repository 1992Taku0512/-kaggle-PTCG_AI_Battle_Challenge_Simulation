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
import collections
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
        self.recent_results = collections.deque(maxlen=self.config.recent_winrate_window)
        
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
        ckpt = torch.load(abs_path, map_location=self.device, weights_only=False)
        
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
        a2_wins = 0
        draws = 0
        winrate = 0.0
        avg_turns = 0.0

        match_wins1 = re.search(r"Agent 1.*?Wins\s+:\s+(\d+)\s+/\s+(\d+)\s+\(([\d.]+)%\)", clean_stdout, re.DOTALL)
        if match_wins1:
            a1_wins = int(match_wins1.group(1))
            winrate = float(match_wins1.group(3))

        match_wins2 = re.search(r"Agent 2.*?Wins\s+:\s+(\d+)", clean_stdout)
        if match_wins2:
            a2_wins = int(match_wins2.group(1))

        match_draws = re.search(r"Draws / Unfinished\s+:\s+(\d+)", clean_stdout)
        if match_draws:
            draws = int(match_draws.group(1))

        match_turns = re.search(r"Average Turn Count\s+:\s+([\d.]+)", clean_stdout)
        if match_turns:
            avg_turns = float(match_turns.group(1))

        return {
            "a1_wins": a1_wins,
            "a2_wins": a2_wins,
            "draws": draws,
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
        p2_mcts = self.mcts_cls(model=opp_model, encoder=self.encoder, num_simulations=self.config.search_count, device=self.device) if isinstance(opp_model, torch.nn.Module) else None

        obs_dict, _ = battle_start(d0, d1)
        turn_count = 0
        max_turns = 150
        episode_experiences = []
        winner = -1

        while obs_dict is not None and turn_count < max_turns:
            turn_count += 1
            if isinstance(obs_dict, dict) and "current" in obs_dict and isinstance(obs_dict["current"], dict):
                res = obs_dict["current"].get("result", -1)
                if res >= 0:
                    winner = res
                    break

            is_done, match_winner = check_match_finish(obs_dict)
            if is_done:
                winner = match_winner
                break

            current_player = extract_active_player(obs_dict)
            obs = to_observation_class(obs_dict)

            if obs.select is None or not obs.select.option:
                action = [0]
            elif current_player == 1 and callable(opp_model):
                # Rule-based agent function (e.g. sampleAgent001)
                try:
                    action = opp_model(obs_dict)
                except Exception as e:
                    num_opts = len(obs.select.option)
                    min_cnt = max(1, getattr(obs.select, 'minCount', 1))
                    action = list(range(min(min_cnt, num_opts)))
            elif current_player == 1 and opp_model is None:
                # RandomAgent opponent action
                num_opts = len(obs.select.option)
                min_cnt = max(1, getattr(obs.select, 'minCount', 1))
                cnt = min(min_cnt, num_opts)
                action = random.sample(list(range(num_opts)), cnt) if num_opts > 0 else [0]
            else:
                state_vec = self.encoder.encode(obs)
                active_mcts = p1_mcts if (current_player == 0 or p2_mcts is None) else p2_mcts
                
                your_deck = d0 if current_player == 0 else d1
                opp_deck = d1 if current_player == 0 else d0

                action_list, _, policy_target = active_mcts.get_action_distribution(obs, your_deck=your_deck, opp_deck=opp_deck)
                num_opts = len(obs.select.option)
                min_cnt = max(1, getattr(obs.select, 'minCount', 1))
                if isinstance(action_list, list) and len(action_list) >= min_cnt:
                    action = action_list
                else:
                    action = list(range(min(min_cnt, num_opts)))

                # Check if encoder supports SparseVector for Transformer models
                sv_dec = self.encoder.encode_decoder(obs, d0 if current_player == 0 else d1) if hasattr(self.encoder, "encode_decoder") else None

                # Step Reward calculation (Side changes, Evolution, Attacks)
                step_reward = 0.0
                if getattr(self.config, "use_intermediate_reward", False):
                    try:
                        p_idx = current_player
                        opp_idx = 1 - current_player
                        p_prizes_curr = getattr(obs, "prize", [3, 3])[p_idx] if hasattr(obs, "prize") and len(obs.prize) > p_idx else 3
                        opp_prizes_curr = getattr(obs, "prize", [3, 3])[opp_idx] if hasattr(obs, "prize") and len(obs.prize) > opp_idx else 3
                        
                        # 1. Side take / lost
                        if hasattr(self, "_prev_prizes"):
                            p_taken = self._prev_prizes[opp_idx] - opp_prizes_curr
                            p_lost = self._prev_prizes[p_idx] - p_prizes_curr
                            if p_taken > 0:
                                step_reward += p_taken * getattr(self.config, "reward_side_take", 0.30)
                            if p_lost > 0:
                                step_reward += p_lost * getattr(self.config, "reward_side_lost", -0.20)
                        self._prev_prizes = {0: getattr(obs, "prize", [3, 3])[0] if hasattr(obs, "prize") else 3,
                                             1: getattr(obs, "prize", [3, 3])[1] if hasattr(obs, "prize") else 3}

                        # 2. Action Specific Shaping (Reward Shaping v7)
                        if obs.select and obs.select.option:
                            ps_curr = obs.player[p_idx] if hasattr(obs, "player") and len(obs.player) > p_idx else None
                            deck_cnt = getattr(ps_curr, "deckCount", 60) if ps_curr else 60

                            for act_i in (action if isinstance(action, list) else [action]):
                                if act_i < len(obs.select.option):
                                    opt = obs.select.option[act_i]
                                    opt_type = getattr(opt, 'type', None)
                                    opt_str = str(opt).lower()

                                    # Attack & Damage Scale Reward (Damage / 1000)
                                    if "attack" in opt_str or opt_type == 12:
                                        dmg = getattr(opt, 'damage', 0)
                                        step_reward += max(0.01, dmg / 1000.0)

                                    # Bench Expansion Bonus (+0.25)
                                    elif "bench" in opt_str or opt_type in (1, 5) or ("play" in opt_str and getattr(obs, "turn", 0) <= 2):
                                        step_reward += 0.25

                                    # Energy Readiness Shaping (+0.05 Active when unready, +0.03 Bench when active ready)
                                    elif "attach" in opt_str or opt_type == 8:
                                        target_inplay = getattr(opt, 'inPlayIndex', 0)
                                        target_energies = getattr(opt, 'energies', [])
                                        if target_inplay == 0:  # Active spot
                                            if len(target_energies) < 3:
                                                step_reward += 0.05  # Active preparation bonus
                                            elif len(target_energies) >= 5:
                                                step_reward -= 0.05  # Over-concentration penalty (>5)
                                        else:  # Bench spot
                                            step_reward += 0.03  # Bench backup preparation bonus

                                    # Danger zone draw penalty (-0.3 when deckCount <= 3)
                                    if deck_cnt <= 3 and ("draw" in opt_str or "play" in opt_str):
                                        step_reward -= 0.30
                    except Exception:
                        pass

                episode_experiences.append({
                    "state": state_vec,
                    "decoder_state": sv_dec,
                    "policy_target": policy_target,
                    "player": current_player,
                    "step_reward": step_reward,
                })

            try:
                obs_dict = battle_select(action)
            except Exception as e:
                import traceback
                print(f"Warning: battle_select failed on turn {turn_count}: {e}")
                traceback.print_exc()
                break

            if isinstance(obs_dict, dict) and "current" in obs_dict and isinstance(obs_dict["current"], dict):
                res = obs_dict["current"].get("result", -1)
                if res >= 0:
                    winner = res
                    break

        if hasattr(self, "_prev_prizes"):
            del self._prev_prizes

        if obs_dict is not None and winner == -1:
            is_done, match_winner = check_match_finish(obs_dict)
            if is_done:
                winner = match_winner

        if not episode_experiences:
            return [], winner

        # TD(lambda) backward smoothing
        time_decay = max(0.5, 1.0 - (turn_count * 0.004))
        
        # Check if loss happened due to Deck-Out (LO) -> Apply heavy penalty (-2.0)
        is_deck_out = False
        if obs_dict and isinstance(obs_dict, dict):
            reason = str(obs_dict.get("reason", "")).lower()
            if "deck" in reason or "lo" in reason:
                is_deck_out = True

        if winner == 0:
            final_value_p0 = 1.0 * time_decay
        elif winner == 1:
            final_value_p0 = -2.0 if is_deck_out else -1.0 * time_decay
        else:
            final_value_p0 = 0.0
        running_value_p0 = final_value_p0
        running_value_p1 = -final_value_p0

        for exp in reversed(episode_experiences):
            player = exp["player"]
            step_r = exp.get("step_reward", 0.0)
            if player == 0:
                target_val = np.clip(running_value_p0 + step_r, -1.0, 1.0)
                running_value_p0 = running_value_p0 * self.config.td_lambda + target_val * (1.0 - self.config.td_lambda)
            else:
                target_val = np.clip(running_value_p1 + (-step_r), -1.0, 1.0)
                running_value_p1 = running_value_p1 * self.config.td_lambda + target_val * (1.0 - self.config.td_lambda)
            exp["value_target"] = target_val

        return episode_experiences, winner

    def update_model(self) -> float:
        """Performs mini-batch SGD update on the PyTorch neural network."""
        if len(self.experience_buffer) < self.config.batch_size:
            return 0.0

        batch_samples = random.sample(self.experience_buffer, self.config.batch_size)
        first_sample = batch_samples[0]

        # 1. SparseVector Transformer Model Update
        if hasattr(first_sample["state"], "index"):
            class LearnInput:
                def __init__(self):
                    self.index = []
                    self.value = []
                    self.offset = []

                def add(self, sv):
                    count = len(self.index)
                    self.index.extend(sv.index)
                    self.value.extend(sv.value)
                    for o in sv.offset:
                        self.offset.append(o + count)

            input_enc = LearnInput()
            input_dec = LearnInput()
            value_labels = []
            policy_labels = []
            masks = []

            for s in batch_samples:
                input_enc.add(s["state"])
                if s["decoder_state"] is not None:
                    input_dec.add(s["decoder_state"])
                value_labels.append(s["value_target"])

                policy = list(s["policy_target"])[:64]
                policy_labels.extend(policy)
                for _ in range(len(policy)):
                    masks.append(1.0)
                num_pad = max(0, 64 - len(policy))
                for _ in range(num_pad):
                    masks.append(0.0)
                    policy_labels.append(0.0)

            idx_enc = torch.tensor(input_enc.index, dtype=torch.int64, device=self.device)
            val_enc = torch.tensor(input_enc.value, dtype=torch.float32, device=self.device)
            off_enc = torch.tensor(input_enc.offset, dtype=torch.int64, device=self.device)

            idx_dec = torch.tensor(input_dec.index, dtype=torch.int64, device=self.device)
            val_dec = torch.tensor(input_dec.value, dtype=torch.float32, device=self.device)
            off_dec = torch.tensor(input_dec.offset, dtype=torch.int64, device=self.device)

            label_v = torch.tensor(value_labels, dtype=torch.float32, device=self.device).unsqueeze(1)
            label_p = torch.tensor(policy_labels, dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
            mask_tensor = torch.tensor(masks, dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)

            self.model.train()
            pred_v, pred_p = self.model(idx_enc, val_enc, off_enc, idx_dec, val_dec, off_dec)

            loss_v = F.huber_loss(pred_v, label_v, delta=0.2)
            loss_p = F.huber_loss(pred_p, label_p, reduction="none", delta=0.1)
            valid_option_count = mask_tensor.sum()
            loss_p = (loss_p * mask_tensor).sum() / (valid_option_count + 1e-8)

            total_loss = loss_v + loss_p

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            return total_loss.item()

        # 2. Standard Dense Vector Model Update
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
        
        # Send Start LINE Notification
        if self.config.use_line_notify:
            start_msg = (
                f"🚀 【学習開始通知】\n"
                f"実験名: {self.config.experiment_name}\n"
                f"• 学習規模: Ep {self.start_episode} -> Ep {self.config.num_episodes}\n"
                f"• デッキ形式: P1({self.config.p1_deck_mode}) vs P2({self.config.p2_deck_mode})\n"
                f"• MCTS探索数: {self.config.search_count}回/ターン\n"
                f"• LINE通知: {self.config.line_notify_every}エピソードごと"
            )
            send_line_notification(start_msg)

        start_time = time.time()
        last_loss = 0.0

        for ep in range(self.start_episode, self.config.num_episodes + 1):
            # 1. Sample decks and opponent dynamically
            (p1_name, p1_deck), (p2_name, p2_deck) = self.deck_provider.sample_deck_pair(
                self.config.p1_deck_mode, self.config.p2_deck_mode
            )
            opp_type = self.opponent_provider.sample_opponent_type()
            opp_desc, opp_model = self.opponent_provider.get_opponent_model(opp_type, self.model)

            # 2. Collect episode experiences & track winner
            ep_samples, winner = self.collect_self_play_episode(p1_deck, p2_deck, opp_model=opp_model)
            if winner == 0:
                self.recent_results.append("WIN")
            elif winner == 1:
                self.recent_results.append("LOSE")
            else:
                self.recent_results.append("DRAW")

            self.experience_buffer.extend(ep_samples)
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

            # 3. Train batch update
            if len(self.experience_buffer) >= self.config.batch_size:
                last_loss = self.update_model()

            n_results = len(self.recent_results)
            win_cnt = self.recent_results.count("WIN")
            lose_cnt = self.recent_results.count("LOSE")
            draw_cnt = self.recent_results.count("DRAW")
            recent_wr = ((win_cnt + draw_cnt * 0.5) / n_results * 100.0) if n_results else 0.0
            breakdown_str = f"WIN: {win_cnt}, LOSE: {lose_cnt}, DRAW: {draw_cnt}"

            # Logging info
            if ep % 50 == 0 or ep == 1:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"[Ep {ep:4d}/{self.config.num_episodes}] Elapsed: {elapsed} | Loss: {last_loss:.4f} | Recent {n_results}G Winrate: {recent_wr:.1f}% ({breakdown_str}) | Decks: P1({p1_name}) vs P2({p2_name}) | Opp: {opp_desc}")

            # 4a. Standalone LINE Notification (no eval required)
            if self.config.use_line_notify and ep % self.config.line_notify_every == 0:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                msg = (
                    f"📊 [{self.config.experiment_name}]\n"
                    f"Episode: {ep}/{self.config.num_episodes}\n"
                    f"• 経過時間: {elapsed}\n"
                    f"• 直近 {n_results} 試合勝率: {recent_wr:.1f}%\n"
                    f"  ({breakdown_str})\n"
                    f"• Loss: {last_loss:.4f}"
                )
                send_line_notification(msg)

            # 4b. Periodic Checkpoint Saving
            if ep % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(ep, is_best=False)
                print(f"💾 Checkpoint saved at Episode {ep}")

        print(f"\n🎉 Training Successfully Completed for {self.config.experiment_name}!")

        # 5. Post-Training Comprehensive Benchmark against Past Agents & Sample (20 Games: 10 First / 10 Second)
        if agent_save_dir:
            print("\n" + "=" * 80)
            print(f"🏁 Running 20-Game Benchmark (10 First / 10 Second) for {self.config.experiment_name}")
            print("=" * 80)

            # Save latest model weights to agent directory before benchmark
            if hasattr(self.model, "state_dict"):
                weights_dst = os.path.join(project_root, agent_save_dir, "model_weights.pt")
                torch.save(self.model.state_dict(), weights_dst)

            past_targets = getattr(self.config, "benchmark_targets", [
                ("Official Sample", "data/sample_submission/sample_submission"),
                ("submit001 Agent", "dev/submit001"),
                ("submit003 Agent", "dev/submit003"),
                ("submit005 Agent", "dev/submit005"),
            ])

            summary_lines = []
            for target_name, target_dir in past_targets:
                if os.path.exists(os.path.join(project_root, target_dir)):
                    res = self.run_eval_subprocess(agent_save_dir, target_dir)
                    line_str = (
                        f"• vs {target_name}: 勝率 {res['winrate']:.1f}%\n"
                        f"  (WIN: {res['a1_wins']}, LOSE: {res['a2_wins']}, DRAW: {res['draws']}) [平均 {res['avg_turns']:.1f}T]"
                    )
                    summary_lines.append(line_str)
                    print(line_str.replace("\n  ", " "))

            final_msg = (
                f"🏁 【{self.config.experiment_name} 完走・20戦対局ベンチマーク結果】\n"
                f"全 {self.config.num_episodes} エピソード学習完了！\n"
                f"• 直近 {n_results} 試合自己対局勝率: {recent_wr:.1f}%\n"
                f"  ({breakdown_str})\n\n"
                f"【対過去モデル/サンプル 20戦成績 (先攻10/後攻10)】\n"
                + "\n".join(summary_lines)
            )
            if self.config.use_line_notify:
                send_line_notification(final_msg)
