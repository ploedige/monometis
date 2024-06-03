import torch
import torchcontrol as toco

from typing import Dict

class HumanController(toco.PolicyModule):
    # stolen from SimulationFramework
    def __init__(self, 
                 robot_model: toco.models.RobotModelPinocchio, 
                #  assistive_gain:torch.Tensor = torch.Tensor([0.26, 0.44, 0.40, 1.11, 1.10, 1.20, 0.85]),
                 assistive_gain:torch.Tensor = torch.Tensor([1.0, 0.1, 1.0, 0.75, 4.0, 1.25, 1.0]),
                 centering_gain:torch.Tensor = torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0])):
        """Controller that assists the moving of a robot by a human

        Args:
            robot_model (toco.models.RobotModelPinocchio): model that physically describes the robot
            assistive_gain (torch.Tensor, optional): gain for assisting the movement of the robot. Defaults to torch.Tensor([0.26, 0.44, 0.40, 1.11, 1.10, 1.20, 0.85]).
            centering_gain (torch.Tensor, optional): this gain lightly forces the robot back into its idle position. Set all elements to 0 to disable. Defaults to torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0]).
        """
        super().__init__()

        joint_angle_limits = robot_model.get_joint_angle_limits()
        self._joint_pos_min = joint_angle_limits[0]
        self._joint_pos_max = joint_angle_limits[1]

        self._assistive_gain = assistive_gain

        self._centering_gain = centering_gain

    def _get_centering_torques(self, joint_pos_current: torch.Tensor) -> torch.Tensor:
        left_boundary = 1 / torch.clamp(torch.abs(self._joint_pos_min - joint_pos_current), 1e-8, 100_000)
        right_boundary = 1 / torch.clamp(torch.abs(self._joint_pos_max - joint_pos_current), 1e-8, 100_000)
        centering_load = left_boundary - right_boundary
        return self._centering_gain * centering_load

    def _get_assistive_torques(self, external_torques: torch.Tensor) -> torch.Tensor:
       return self._assistive_gain * external_torques

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ### ATTENTION: state_dict["motor_torques_external"] are exactly opposite to the expectation
        motor_torques_external = -state_dict["motor_torques_external"] #correct external torques direction
        assistive_torques = self._get_assistive_torques(motor_torques_external)
        centering_torques = self._get_centering_torques(state_dict["joint_positions"])

        return {"joint_torques": assistive_torques + centering_torques}
