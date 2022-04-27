# pylint: skip-file
import gym
import numpy as np
import math
import pybullet as p
from cogment_verse_environment.pybullet_driving_env.resources.car import Car
from cogment_verse_environment.pybullet_driving_env.resources.plane import Plane
from cogment_verse_environment.pybullet_driving_env.resources.goal import Goal
from copy import deepcopy
import time

# import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -0.6], dtype=np.float32), high=np.array([1, 0.6], dtype=np.float32)
        )

        """
            obs_space -- observation space for the environment
                car_qpos     -- x and y co-ordinates of the car, orientation of the car (Euler angles/pi), x and y velocities of the car
                segmentation -- A 3d occupancy map with Region occupied by the car along the first layer, that occupied by the obstacles along the second layer,
                                and the goal position along the third layer (not present for the agent setting the goal)
        """
        obs_space = {
            "car_qpos": gym.spaces.box.Box(
                low=np.array([-np.Inf, -np.Inf, -1, -1, -1, -np.Inf, -np.Inf], dtype=np.float32),
                high=np.array([np.Inf, np.Inf, 1, 1, 1, np.Inf, np.Inf], dtype=np.float32),
            ),
            "segmentation": gym.spaces.box.Box(
                low=np.zeros((75, 75, 3)),
                high=np.ones((75, 75, 3)),
            ),
        }

        self.observation_space = gym.spaces.dict.Dict(obs_space)
        self.np_random, _ = gym.utils.seeding.np_random()

        # use this if not rendering
        self.client = p.connect(p.DIRECT)

        # use this if rendering
        # self.client = p.connect(p.GUI)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1 / 30, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.max_steps = 250
        self.steps = 0

        # initialize the obstacles and their dimentions
        self.pos_obstacles = None
        self.obstacle_dims = None
        self.initialize_obstacles()

        self.obstacle_mass = 1000

        # setting up the camera
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 25], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 1, 0]
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=50.0, aspect=1.0, nearVal=0.1, farVal=100.5)
        self.threshold = 1

    def initialize_obstacles(self):
        # TODO: once in running condition, change the position initialization to np.zeros
        """
        Write your own initialization
        pos_onstacles changes every time reset for alice is called
        """
        # TODO: once in running condition, change the position initialization to np.zeros
        self.pos_obstacles = np.array(
            [
                [7, 7, 0.1],
                [-3, 4, 0.1],
                [5, -6, 0.1],
                [-7, -5, 0.1],
                [1, -2, 0.1],
                # positions of outer walls (constant throughout: see reset_obstacle_positions)
                [0, 12, 0.1],
                [0, -12, 0.1],
                [12, 0, 0.1],
                [-12, 0, 0.1],
            ]
        )

        wall_width = 11
        self.obstacle_dims = np.array(
            [
                # obstacles
                [1, 1, 0.2],
                [1, 1, 0.2],
                [1, 1, 0.2],
                [1, 1, 0.2],
                [1, 1, 0.2],
                # outer walls
                [wall_width, 1, 0.2],
                [wall_width, 1, 0.2],
                [1, wall_width, 0.2],
                [1, wall_width, 0.2],
            ]
        )

    def get_camera_image(self, sz):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=sz, height=sz, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix
        )
        # plt.imshow(segImg); plt.show()
        return width, height, rgbImg, depthImg, segImg

    def get_observation_image(self, sz, agent="alice"):
        if agent == "alice":
            w, h, _, _, seg = self.get_camera_image(sz)
            obs = (seg > 1).astype(float)
            car = (seg == 1).astype(float)
            gridmap = np.dstack((car, obs))
        else:
            w, h, _, _, seg = self.get_camera_image(sz)
            obs = (seg > 2).astype(float)
            car = (seg == 1).astype(float)
            goal = (seg == 2).astype(float)
            gridmap = np.dstack((car, obs, goal))
        return gridmap

    def draw_obstacles(self):
        for i in range(len(self.pos_obstacles)):
            boxHalfLength = self.obstacle_dims[i][0]
            boxHalfWidth = self.obstacle_dims[i][1]
            boxHalfHeight = self.obstacle_dims[i][2]
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
            p.createMultiBody(
                baseMass=self.obstacle_mass, baseCollisionShapeIndex=colBoxId, basePosition=self.pos_obstacles[i, :]
            )

    def step(self, action, max_step_multiplier=1, agent="alice"):
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        dist_to_goal = np.linalg.norm(car_ob[:2] - self.goal)
        self.prev_dist_to_goal = dist_to_goal

        # Reward function
        # Modify the reward function according to your needs
        reward = 0
        self.steps += 1
        if dist_to_goal < self.threshold:
            self.done = True
            reward = 5
        if self.steps > max_step_multiplier * self.max_steps:
            reward = -2
            self.done = True

        ob = np.array(car_ob, dtype=np.float32)
        gridmap = self.get_observation_image(75, agent)
        return {"segmentation": gridmap, "car_qpos": ob}, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, goal, base_position, base_orientation, agent="alice"):
        self.steps = 0
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client, base_position, base_orientation)

        self.goal = np.array(deepcopy(goal))
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 + (car_ob[1] - self.goal[1]) ** 2))

        if agent == "alice":
            # If agent is alice, reset obstacle positions (since we want bob to run through same obstacles as alice)
            self.reset_obstacle_positions()

        self.draw_obstacles()
        gridmap = self.get_observation_image(75, agent)
        # print("Reset time: ", time.time() - tic)
        return {"segmentation": gridmap, "car_qpos": np.array(car_ob, dtype=np.float32)}

    def reset_obstacle_positions(self):
        """
        reset the obstacle positions based on some probability distribution
        Write your own reset function
        """
        self.pos_obstacles[0][0:2] = np.random.uniform(-5, 5, 2)
        mult = -1
        for i in range(2):
            mult = -1 * mult
            r = np.random.randint(0, 3)
            if r == 0:
                self.pos_obstacles[i][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([0, 5]))
            elif r == 1:
                self.pos_obstacles[i][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([5, 5]))
            elif r == 2:
                self.pos_obstacles[i][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([5, 0]))
        mult = -1
        for i in range(2):
            mult = -1 * mult
            r = np.random.randint(0, 3)
            if r == 0:
                self.pos_obstacles[i + 2][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([-5, 5]))
            elif r == 1:
                self.pos_obstacles[i + 2][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([-10, 5]))
            elif r == 2:
                self.pos_obstacles[i + 2][0:2] = mult * (np.random.uniform(0, 5, 2) + np.array([-10, 0]))

    def close(self):
        p.disconnect(self.client)
