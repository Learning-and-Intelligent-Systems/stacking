"""
Description of entities for 2D throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

class ThrowingBall:
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


class ThrowingAction:
    """ Represents a throwing action """
    def __init__(self, obj, init_pos=[0,0,0], init_vel=[0,0,0]):
        self.object = obj
        self.x, self.y, self.th = init_pos
        self.vx, self.vy, self.w = init_vel
