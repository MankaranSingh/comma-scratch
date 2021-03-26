import numpy as np

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        self.first = True
        
        self.L = 10
        self.lr = self.L / 2 + 1
        self.previous_time = 0
        
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
    def step(self, v, delta, current_time, a_x=0, a_y=0):
        
        if self.first:
            dt = 0.049
            self.first = False
        else:
            dt = current_time - self.previous_time
        dx = v*np.cos(self.theta) * dt + 0.5 * a_x * dt ** 2
        dy = v*np.sin(self.theta) * dt + 0.5 * a_y * dt ** 2
        self.xc = self.xc + dx
        self.yc = self.yc + dy
        self.theta = self.theta + v*np.cos(self.beta)*np.tan(self.delta)/self.L * dt
        #self.delta = np.deg2rad(delta)
        self.delta = np.clip(delta, -0.005, 0.005)
        self.beta = np.arctan(self.lr*self.delta/self.L)
        self.previous_time = current_time

        return dx, dy
