from abc import ABC, abstractmethod
from torch import Tensor

from core.specs import TimedActionTrajectory, Observation, Reward, State

class RewardModel(ABC):
    components: list[str]

    @abstractmethod
    def __call__(
        self,
        states: State,
        observations: Observation,
        actions: TimedActionTrajectory,
    ) -> tuple[Reward, dict[str, Reward]]:
        pass
