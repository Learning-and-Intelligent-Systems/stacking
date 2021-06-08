"""
Description of ball for 2D throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

class Ball:
    color = [1,0,0]
    mass = 1.0
    radius = 0.025
    air_drag_linear = 1
    air_drag_angular = 1e-5
    friction_coef = 0.8
    rolling_resistance = 1e-4
    bounciness = 0.5

