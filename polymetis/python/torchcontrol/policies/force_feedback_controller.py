from typing import Dict

import torch
import torchcontrol as toco
import numpy as np

from .human_controller import HumanController

class ForceFeedbackController(HumanController):
    def __init__(self, 
                 robot_model: toco.models.RobotModelPinocchio, 
                 assistive_gain:torch.Tensor = torch.Tensor([0.26, 0.44, 0.40, 1.11, 1.10, 1.20, 0.85]),
                 centering_gain:torch.Tensor = torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0]),
                 initial_replication_torques: torch.Tensor = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                 force_feedback_damping_gain: torch.Tensor = torch.Tensor([25.0, 25.0, 25.0, 25.0, 7.5, 4.0, 4.0]),
                 force_feedback:bool = True):
        """HumanController with capability to give force feedback to the user
        To reproduce a torque the torque has to be transmitted to this controller via the update_parameter({"replication_torque": <torques_to_be_reproduced>}) function

        Args:
            robot_model (toco.models.RobotModelPinocchio): physical model of the robot
            assistive_gain (torch.Tensor, optional): gain for assisting the movement of the robot. Defaults to torch.Tensor([0.26, 0.44, 0.40, 1.11, 1.10, 1.20, 0.85]).
            centering_gain (torch.Tensor, optional): this gain lightly forces the robot back into its idle position. Set all elements to 0 to disable. Defaults to torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0]).
            initial_replication_torques (torch.Tensor, optional): initialize the force feedback. Defaults to torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).
            force_feedback_damping_gain (torch.Tensor, optional): gain for damping the force feedback depending on the current velocity. Defaults to torch.Tensor([25.0, 25.0, 25.0, 25.0, 7.5, 4.0, 4.0]).
            force_feedback (bool, optional): enables/disables force feedback. Defaults to True.
        """
        super().__init__(robot_model, assistive_gain, centering_gain)
        self.replication_torques = torch.nn.Parameter(initial_replication_torques)
        self._force_feedback_damping_gain = force_feedback_damping_gain
        self._force_feedback = force_feedback

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ### ATTENTION: state_dict["motor_torques_external"] are exactly opposite to the expectation
        motor_torques_external = -state_dict["motor_torques_external"] #correct external torques direction
        joint_pos = state_dict["joint_positions"]
        assistive_torques = self._get_assistive_torques(motor_torques_external)
        centering_torques = self._get_centering_torques(joint_pos)
        force_feedback_torques = self.replication_torques            

        if self._force_feedback:
            return_torques = torch.zeros_like(motor_torques_external)
            for idx in range(return_torques.size()[0]):
                demonstrator_torque = motor_torques_external[idx]
                feedback_torque = force_feedback_torques[idx]
                if demonstrator_torque * feedback_torque < 0: #torques in different directions
                    if abs(demonstrator_torque) <= abs(feedback_torque):
                        return_torques[idx] = -demonstrator_torque
                    else:
                        return_torques[idx] = feedback_torque
            return {"joint_torques": return_torques}
        else:
            return {"joint_torques": assistive_torques + centering_torques}
