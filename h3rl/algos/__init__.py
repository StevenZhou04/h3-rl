"""RL algorithms. Each is a class registered by name; the trainer (h3rl/train.py) does everything else."""
from h3rl.algos.base import ALGORITHMS, Algorithm, register
from h3rl.algos import nft, grpo   # noqa: F401  (registration side effects)
