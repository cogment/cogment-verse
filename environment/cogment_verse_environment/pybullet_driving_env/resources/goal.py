import pybullet as p
import os


class Goal:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)


