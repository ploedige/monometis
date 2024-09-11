import torch
from typing import Dict

import torchcontrol as toco
from torchcontrol.utils import to_tensor

class ExtendedImpedanceTrajectoryExecutor(toco.PolicyModule):
    """
    Adaptation of the JointTrajectoryExecutor that allows changing the trajectory during runtime 
    """
    def __init__(
            self,
            ee_traj_desired: torch.Tensor,
            torque_profile_desired: torch.Tensor,
            Kq: torch.Tensor,
            Kqd: torch.Tensor,
            Kx: torch.Tensor,
            Kxd: torch.Tensor,
            robot_model: torch.nn.Module,
            ignore_gravity=True,
    ):
        """
        Executes an end-effector based trajectory using a joint PD controller while allowing to switch the trajectory during execution

        Args:
            init_ee_traj_desired (torch.Tensor): end-effector pose trajectory as tensor of shape (nS,7) (7d pose = 3d position + 4d orientation (quaternion))
            Kq: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kqd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kx: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kxd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension, nS the number of steps and N is the number of degrees of freedom)
        """
        super().__init__()
        # initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        self.ee_traj_desired = ee_traj_desired
        self.torque_profile_desired = torque_profile_desired

        # step count
        self.step = 0
            
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # current robot state
        joint_pos_current = state_dict['joint_positions']
        joint_vel_current = state_dict['joint_velocities']

        # increment step and check if finished
        self.step += 1
        if self.step >= self.ee_traj_desired.shape[0]:
            self.set_terminated()
            joint_pos_desired = joint_pos_current
            torque_desired = torch.tensor([0] * 7)
        else:
            # inverse kinematics
            ee_pos_desired = self.ee_traj_desired[self.step,:3]
            ee_quat_desired = self.ee_traj_desired[self.step,3:]
            joint_pos_desired = self.robot_model.inverse_kinematics(
                ee_pos_desired, ee_quat_desired, rest_pose=joint_pos_current
            )
            torque_desired = self.torque_profile_desired[self.step]

        joint_vel_desired = torch.zeros_like(joint_vel_current)
        # control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            joint_pos_desired,
            joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )# coriolis

        torque_out = torque_feedback + torque_desired + torque_feedforward

        return {"joint_torques": torque_out}
