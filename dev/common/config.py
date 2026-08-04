from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainerConfig:
    """Configuration data structure for PTCGTrainer."""
    experiment_name: str = "submit006_run"
    
    # 1. Model initialization & Resume mode
    resume_checkpoint_path: Optional[str] = None  # If set, loads weights & state to resume training
    device: str = "cuda"  # "cuda" or "cpu"
    
    # 2. Random Opponent Mode settings
    # Options: "self_play", "random", "past_checkpoint"
    opponent_types: List[str] = field(default_factory=lambda: ["self_play", "random", "past_checkpoint"])
    opponent_weights: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    past_checkpoint_dir: Optional[str] = None  # Directory containing past model .pt files
    
    # 3. Random Deck Mode settings
    deck_pool_paths: List[str] = field(default_factory=list)  # List of paths to candidate deck CSVs
    p1_deck_mode: str = "random"  # "random" or "fixed"
    p2_deck_mode: str = "random"  # "random" or "fixed"
    
    # 4. Training Hyperparameters
    num_episodes: int = 10000
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    td_lambda: float = 0.9
    search_count: int = 10  # MCTS search iterations during training
    
    # 5. Evaluation & Checkpoint & Notification settings
    eval_every: int = 500
    eval_num_games: int = 20
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_every: int = 500
    
    use_line_notify: bool = True
    line_notify_every: int = 500
    recent_winrate_window: int = 200  # Sliding window size for recent winrate calculation
