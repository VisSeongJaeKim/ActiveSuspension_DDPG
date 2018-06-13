import numpy as np
from scipy import signal
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import pdb

class ActiveSuspension(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.mb = 300.0
        self.mw = 60.0
        self.bs = 1000     #1000
        self.ks = 16000.0    #16000
        self.kt = 190000.0   #190000

        self.A = np.array([[0.0, 1.0, 0.0, 0.0], [-self.ks / self.mb, -self.bs / self.mb, self.ks / self.mb, self.bs / self.mb],
             [0.0, 0.0, 0.0, 1.0], [self.ks / self.mw, self.bs / self.mw, (-self.ks - self.kt) / self.mw, -self.bs / self.mw]])
        self.B = np.array([[0.0, 0.0], [0.0, 1000.0 / self.mb], [0.0, 0.0], [self.kt / self.mw, -1000.0 / self.mw]])

        self.C = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0],
             [-self.ks / self.mb, -self.bs / self.mb, self.ks / self.mb, self.bs / self.mb]])
        self.D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1000.0 / self.mb]])
        self.dt_requested=0.5
        self.A, self.B, self.C, self.D, self.dt = signal.cont2discrete((self.A, self.B, self.C, self.D), self.dt_requested)

        self.i = 0
        self.max_force_mag = 0.8
        self.U = []

        self.plot_t = []
        self.plot_x = []

        self.min_position = np.array([
            -0.1,
            np.finfo(np.float32).min,
            np.finfo(np.float32).min,
            np.finfo(np.float32).min])

        self.max_position = np.array([
            0.1,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.low = np.array([self.min_position])
        self.high = np.array([self.max_position])

        self.viewer = None
        self.state = (0.0,0.0,0.0,0.0)

        self.action_space = spaces.Box(low=-self.max_force_mag, high=self.max_force_mag, shape=(1,))
        self.observation_space = spaces.Box(self.min_position, self.max_position)
        self.done = False

        self.seed()
        self.reset()

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, force):
        x1, x1dot, x2, x2dot = self.state

        X = np.array([x1, x1dot,x2, x2dot])
        #print(x1)
        self.plot_x.append(x1)
        force = np.clip(force, -self.max_force_mag, self.max_force_mag)[0]
        # if self.i > 25:
        #     self.U = np.array([0.0 * (1 - np.cos(0.6 * self.i)), force])
        # else:
        self.U = np.array([0.025 * (1 - np.cos(0.6 * self.i)), force])


        self.plot_t.append(self.i)
        self.i = self.i+1

        X = np.matmul(self.A, X) + np.matmul(self.B, self.U)

        x1 = X[0]
        x1dot = X[1]
        x2 = X[2]
        x2dot = X[3]
        self.state = np.array([x1,x1dot,x2,x2dot])
        costs = (x1**2 + 0.1*x1dot**2 + 0.001*force**2)*1000 #exp(10x1)
        if self.i > 100:
            self.done = True

        return np.array(self.state), -costs, self.done, {}

    def reset(self):
        self.state = (0.0,0.0,0.0,0.0)
        self.steps_beyond_done = None
        self.i=0
        self.U=[]
        self.plot_t = []
        self.plot_x = []
        self.done =False
        return np.array(self.state)

    def close(self):
        if self.viewer: self.viewer.close()

