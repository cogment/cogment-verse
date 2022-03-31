# pylint: skip-file
import pybullet as p
import os
import numpy as np


class Car:
    def __init__(self, client, base_position, base_orientation):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "simplecar.urdf")
        self.car = p.loadURDF(fileName=f_name, basePosition=[0, 0, 0.1], physicsClientId=client)
        # Joint indices as found by p.getJointInfo()
        self.steering_joints = [0, 2]
        self.drive_joints = [1, 3, 4, 5]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20
        base_orientation = p.getEulerFromQuaternion(base_orientation)
        base_orientation = (0, 0, base_orientation[2])
        base_orientation = p.getQuaternionFromEuler(base_orientation)
        p.resetBasePositionAndOrientation(self.car, base_position, base_orientation)

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Set the steering joint positions
        p.setJointMotorControlArray(
            self.car,
            self.steering_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[steering_angle] * 2,
            physicsClientId=self.client,
        )

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag + self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1 / 30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client,
        )

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ang = (ang[0] / np.pi, ang[1] / np.pi, ang[2] / np.pi)
        pos = pos[:2]
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = pos + ang + vel
        return observation
