""" Parameters of the task, that has to be flown by the glider. Describe task?

    Classes
    -------

    TaskParameters
         Task related parameters
"""

import numpy as np


class TaskParameters:
    """ Parameters which are used to describe the soaring task

    Attributes
    ----------

    ORIENTATION: int
        Triangle orientation: alpha - pi/2 ('alpha' referred to in GPS triangle regulations)

    G_T_T: ndarray
        Transformation matrix from triangle-coordinates to local NE-coordinates

    TRIANGLE: ndarray
        Local NE-coordinates of the triangle vertices (2 x 3 array with vertices as columns)

    FINISH_LINE: ndarray
        Local NE-coordinates of the triangle origin, i.e. finish line (2 x 1)

    ONE_T_T: ndarray
        Rotation matrix from triangle-coordinates to sector-one-coordinates

    TWO_T_T: ndarray
        Rotation matrix from triangle-coordinates to sector-two-coordinates

    THREE_T_T: ndarray
        Rotation matrix from triangle-coordinates to sector-two-coordinates

    TASK: str
        'Distance' or 'speed' task

    WORKING_TIME: int
        Time to have the task done (relevant for 'distance' only)
    """

    def __init__(self):
        self.ORIENTATION = 0
        self.G_T_T = np.transpose(np.array([[np.cos(self.ORIENTATION), np.sin(self.ORIENTATION)],
                                            [-np.sin(self.ORIENTATION), np.cos(self.ORIENTATION)]]))

        self.TRIANGLE = self.G_T_T @ np.array([[0., 350, 0.],
                                               [350, 0., -350]])

        self.FINISH_LINE = self.TRIANGLE[:, 2] + (self.TRIANGLE[:, 0] - self.TRIANGLE[:, 2]) / 2

        self.ONE_T_T = np.array([[np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8)],
                                 [-np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8)]])

        self.TWO_T_T = np.array([[np.cos(-np.pi / 4), np.sin(-np.pi / 4)],
                                 [-np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])

        self.THREE_T_T = np.array([[np.cos(9 * np.pi / 8), np.sin(9 * np.pi / 8)],
                                   [-np.sin(9 * np.pi / 8), np.cos(9 * np.pi / 8)]])

        self.TASK = 'distance'
        self.WORKING_TIME = 60 * 30
