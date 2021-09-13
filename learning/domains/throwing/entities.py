"""
Description of entities for 2D throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""
import numpy as np
import scipy

def truncnorm(lower, upper, mu, sigma, size=(1,)):
    X = scipy.stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(size)

class ThrowingBall:
    dim = 10 # vectorized dimensions

    """ Represents a throwable ball with visual and physical properties """
    def __init__(self, color=[1,0,0], mass=1.0, radius=0.025,
                 air_drag_linear=1, air_drag_angular=1e-5,
                 friction_coef=0.8, rolling_resistance=1e-4,
                 bounciness=0.5):

        # Visual properties
        self.color = color

        # Inertial properties
        self.mass = mass
        self.radius = radius
        self.inertia = 0.25 * self.mass * self.radius**2

        # Air drag
        self.air_drag_linear = air_drag_linear
        self.air_drag_angular = air_drag_angular

        # Friction, rolling, and bounce
        self.friction_coef = friction_coef
        self.rolling_resistance = rolling_resistance
        self.bounciness = bounciness


    @staticmethod
    def random():
        color = np.random.rand(3)
        mass = np.random.uniform(0.5, 1.5)
        radius = np.random.uniform(0.02, 0.06)
        air_drag_linear = np.random.uniform(0, 2)
        air_drag_angular = np.random.uniform(5e-6, 5e-5)
        friction_coef = np.random.uniform(0.1, 1)
        rolling_resistance = np.random.uniform(1e-4, 1e-3)
        bounciness = np.random.uniform(0.1, 0.6)

        return ThrowingBall(mass=mass, radius=radius)

    def vectorize(self):
        v = np.zeros(self.dim)
        v[:3] = self.color
        v[3] = self.mass
        v[4] = self.radius
        v[5] = self.air_drag_linear
        v[6] = self.air_drag_angular
        v[7] = self.friction_coef
        v[8] = self.rolling_resistance
        v[9] = self.bounciness
        return v

    @staticmethod
    def from_vector(v):
        return ThrowingBall(color=v[:3],
                            mass=v[3],
                            radius=v[4],
                            air_drag_linear=v[5],
                            air_drag_angular=v[6],
                            friction_coef=v[7],
                            rolling_resistance=v[8],
                            bounciness=v[9])

class ThrowingAction:
    dim = 2 # vectorized dimensions

    """ Represents a throwing action """
    def __init__(self, obj, init_pos=[0,0,0], init_vel=[0,0,0]):
        self.object = obj
        self.x, self.y, self.th = init_pos
        self.vx, self.vy, self.w = init_vel

    @staticmethod
    def from_vector(b, vec):
        ang, w = vec
        vel = 5
        init_vel = [
            vel * np.cos(ang),
            vel * np.sin(ang),
            w,
        ]
        return ThrowingAction(b, init_pos=[0,0.25,0], init_vel=init_vel)

    @staticmethod
    def random_vector(n_samples=0):
        ang = np.random.uniform(np.pi/8, 3*np.pi/8, size=max(n_samples, 1))
        w = np.random.uniform(-10, 10, size=max(n_samples, 1))
        if n_samples == 0:
            return [ang, w]
        else:
            return np.stack([ang, w], axis=1)
