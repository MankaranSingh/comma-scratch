from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: 'mean={:.5f}, var={:.5f}'.format(s[0], s[1])

class KalmanFilter():
    def __init__(self, init_x, init_var, R, Q):
        self.x = init_x  # state
        self.P = init_var   # state variance
        self.R = R          # sensor variance
        self.Q = Q          # process variance
        self.K = 0
         
    def update(self, z):
        y = z - self.x
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * y
        self.P = (1 - self.K) * self.P
        
    def predict(self, dx):
        self.x = self.x + dx
        self.P = self.P + self.Q
