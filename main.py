import gymnasium as gym
from models import ModelType

frozen_lake_env = gym.make(
    ModelType.FROZEN_LAKE,
    desc=None,
    map_name='4x4',
    is_slippery=True
)
