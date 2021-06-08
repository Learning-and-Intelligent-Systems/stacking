"""
Agent for learning in the 2D throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from learning.domains.throwing.ball import Ball

# Constants
gravity = 9.80665
bounce_vel_thresh = -0.05

class ThrowingAgent:

    def __init__(self, objects=[]):
        # Simulation parameters
        self.dt = 0.001
        self.tmax = 5

        # TODO: Initialize a set of ball objects
        self.objects = objects


    def sample_action(self):
        """
        Samples an action, which consists of a ball and release velocities
        """

        # Sample a random ball
        b = np.random.choice(self.objects)

        # Sample random velocities
        v = 5
        ang = np.random.uniform(np.pi/8, 3*np.pi/8)
        w = np.random.uniform(-10, 10)

        # Package up the action
        init_vel = [
            v * np.cos(ang),
            v * np.sin(ang),
            w, 
        ]
        return b, init_vel


    def simulate(self, action, do_animate=False, do_plot=False):
        """ Simulate an action """
        
        tvec = np.arange(0, self.tmax+self.dt, self.dt)
        num_pts = tvec.shape[0]

        # Choose a ball
        b = action[0]

        # Set initial state
        # The format is [x, y, theta, xdot, ydot, thetadot]
        state = np.zeros((6, num_pts))
        state[:,0] = [
            0,
            0,
            0,
            action[1][0],
            action[1][1],
            action[1][2],
        ]
        in_air = True
        
        # TODO: Finish episode prematurely if needed
        for i in range(tvec.shape[0]-1):
            
            x, y, theta, vx, vy, w = state[:,i]

            # Calculate all the accelerations
            forces = np.array([0.0, 0.0, 0.0])
            
            # Gravity force
            forces += [
                0,
                -b.mass * gravity,
                0,
            ]

            # Air resistance (if the ball is in the air)
            if in_air:
                forces += [
                    -b.air_drag_linear * vx,
                    -b.air_drag_linear * vy,
                    -b.air_drag_angular * w,
                ]

            # Friction and rolling resistance (if the ball is on the ground)
            if not in_air:
                v_ground = vx + b.radius*w
                f_friction = b.friction_coef * b.mass * gravity
                forces += [
                    -np.sign(v_ground) * f_friction, 
                    0,
                    -b.rolling_resistance * w,
                ] 

            # Propagate state
            state_dot = np.array([
                vx,
                vy,
                w,
                forces[0] / b.mass,
                forces[1] / b.mass,
                forces[2] / (0.25 * b.mass * b.radius**2),
            ])
            state[:,i+1] = state[:,i] + self.dt*state_dot

            # Bounce/collision dynamics:
            # If the height is less than zero, force it to zero.
            # Also, correct velocity by bouncing if past a specified velocity 
            # threshold, else setting the vertical velocity to zero.
            if state[1,i+1] < 0:
                state[1,i+1] = 0
                new_vy = state[4,i+1]
                if new_vy < bounce_vel_thresh:
                    state[4,i+1] *= -b.bounciness
                    in_air = True
                else:
                    state[4,i+1] = 0
                    in_air = False

        # Visualize
        if do_animate:
            min_x = min(state[0,:])
            max_x = max(state[0,:])
            min_y = min(state[1,:])
            max_y = max(state[1,:])
            xrange = max_x - min_x
            yrange = max_y - min_y
            xbuf = 0.1*xrange
            ybuf = 0.1*yrange

            fig = plt.figure()
            ax = fig.add_subplot(111, aspect="equal")
            xdata, ydata = [], []
            line, = ax.plot([], [], "b-")
            circ = plt.Circle((0,0), b.radius, color=b.color)
            ori_line, = ax.plot([0,b.radius],[0,0], color="k")

            def init():
                ax.set_xlim(min_x-xbuf, max_x+xbuf)
                ax.set_ylim(min_y-ybuf, max_y+ybuf)
                ax.add_patch(circ)
                return line, circ, ori_line

            def animate(i):
                x = state[0,i]
                y = state[1,i]
                th = state[2,i]
                xdata.append(x)  # update the data.
                ydata.append(y)  # update the data.
                circ.center = x, y
                line.set_data(xdata, ydata)

                x_ori = [x, x+b.radius*np.cos(th)]
                y_ori = [y, y+b.radius*np.sin(th)]
                ori_line.set_data(x_ori, y_ori)
                return line, circ, ori_line

            ani = animation.FuncAnimation(
                fig, animate, frames=num_pts, 
                init_func=init, interval=1, blit=True, repeat=False)
            plt.show()
            print("Done")


        if do_plot:
            plt.figure()
            x = state[0,:]
            y = state[1,:]
            l = plt.plot(x, y)
            plt.show()
            


if __name__=="__main__":

    b1 = Ball()
    b1.bounciness = 0.85

    b2 = Ball()
    b2.mass = 1.2
    b2.color = [0,0,1]
    b2.radius = 0.03
    b2.air_drag_angular = 1e-5
    b2.friction_coef = 0.85
    b2.bounciness = 0.4
    b2.rolling_resistance = 1e-4

    b3 = Ball()
    b3.mass = 0.7
    b3.color = [0,0.6,0]
    b3.radius = 0.05
    b3.friction_coef = 0.2
    b3.bounciness = 0.2
    b3.rolling_resistance = 5e-3

    agent = ThrowingAgent([b1,b2,b3])

    for i in range(10):
        act = agent.sample_action()
        agent.simulate(act, do_animate=True)
        