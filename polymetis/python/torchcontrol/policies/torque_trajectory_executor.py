from typing import Dict, List

import torch
import torchcontrol as toco
from torchcontrol.utils.tensor_utils import stack_trajectory, to_tensor

class TorqueTrajectoryExecutor(toco.PolicyModule):
    def __init__(
        self,
        joint_torque_trajectory: List[torch.Tensor],
    ):
        """Executes a torque trajectory

        Args:
            joint_torque_trajectory (List[torch.Tensor]): the torque trajectory to be executed  
        """
        super().__init__()

        self.joint_torque_trajectory = to_tensor(stack_trajectory(joint_torque_trajectory))

        self.N = self.joint_torque_trajectory.size(0)
        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Parse current state
        joint_torque_current = state_dict["motor_torques_measured"]

        # Query plan for desired state
        joint_torque_desired = self.joint_torque_trajectory[self.i, :]

        # Increment & termination
        self.i += 1
        if self.i == self.N:
            self.set_terminated()

        return {"joint_torques": joint_torque_desired}
