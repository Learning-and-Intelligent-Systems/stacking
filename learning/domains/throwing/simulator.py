"""
Ball throwing simulator

Copyright 2021 Massachusetts Institute of Technology
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ThrowingSimulator:
    # Constants
    gravity = 9.80665           # Acceleration due to gravity [m/s^2]
    bounce_vel_thresh = -0.05   # Threshold velocity before stopping bouncing [m/s]
    stop_vel_thresh = 0.01      # Maximum linear velocity before stopping simulation [m/s]
    stop_vel_count = 10         # Number of consecutive counts below velocity threshold before simulation is stopped


    def __init__(self, objects, dt=0.0005, tmax=5):
        """ Initialize the simulator """
        # List of throwable objects
        self.objects = objects

        # Simulation time parameters
        self.dt = dt
        self.tmax = tmax


    def simulate(self, action, do_animate=False, do_plot=False):
        """ Simulate an action """
        # Initialize bookkeeping variables
        num_bounces = 0
        stop_count = 0
        tvec = np.arange(0, self.tmax + self.dt, self.dt)
        num_pts = tvec.shape[0]
        state = np.zeros((6, num_pts))
        in_air = True

        # Unpack the action and set initial state
        # The format is [x, y, theta, xdot, ydot, thetadot]
        b = action.object
        state[:,0] = [
            action.x,
            action.y,
            action.th,
            action.vx,
            action.vy,
            action.w,
        ]

        # MAIN SIMULATION LOOP
        weight = b.mass * self.gravity
        for i in range(num_pts-1):
            x, y, theta, vx, vy, w = state[:,i]
            forces = np.array([0.0, 0.0, 0.0])

            # Gravity force
            forces += [
                0,
                -weight,
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
                f_friction = b.friction_coef * weight
                forces += [
                    -np.sign(v_ground) * f_friction,
                    0,
                    -b.rolling_resistance * w,
                ]

            # Propagate state with Newton's 2nd law and first-order integration step
            state_dot = np.array([
                vx,
                vy,
                w,
                forces[0] / b.mass,
                forces[1] / b.mass,
                forces[2] / b.inertia,
            ])
            state[:,i+1] = state[:,i] + self.dt*state_dot

            # Bounce/collision dynamics:
            # If the height is less than zero, force it to zero.
            # Also, correct velocity by bouncing if past a specified velocity
            # threshold, else setting the vertical velocity to zero.
            if state[1,i+1] < 0:
                state[1,i+1] = 0
                new_vy = state[4,i+1]
                if new_vy < self.bounce_vel_thresh:
                    state[4,i+1] *= -b.bounciness
                    in_air = True
                    num_bounces += 1
                else:
                    state[4,i+1] = 0
                    in_air = False

            # Check stopping condition based on min velocity
            new_vx = state[3,i+1]
            if abs(new_vx) <= self.stop_vel_thresh:
                stop_count += 1
                if stop_count >= self.stop_vel_count:
                    break
            else:
                stop_count = 0

        # Trim the results based on stopping condition
        final_idx = i+1
        tvec = tvec[:final_idx+1]
        state = state[:, :final_idx+1]

        # Visualize/animate if specified
        if do_animate:
            self.animate_results(b, state)
        elif do_plot:
            self.plot_results(b, state)

        # Package results
        results = {
            "time": tvec,
            "state": state,
            "num_bounces": num_bounces,
        }
        return results


    def plot_results(self, obj, state, ax=None, label=None):
        """ Plots simulation trajectory """
        if ax is None:
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)

        plt.title("Throwing Trajectory")

        x = state[0,:]
        y = state[1,:]
        th = state[2,:]
        l = plt.plot(x, y, "-", label=label)

        circ_init = plt.Circle((x[0], y[0]), obj.radius, color=obj.color)
        circ_final = plt.Circle((x[-1], y[-1]), obj.radius, color=obj.color)
        ax.add_patch(circ_init)
        ax.add_patch(circ_final)

        ori_line_init, = ax.plot([x[0], x[0] + obj.radius*np.cos(th[0])],
                                 [y[0], y[0] + obj.radius*np.sin(th[0])],
                                 color="k")
        ori_line_final, = ax.plot([x[-1], x[-1] + obj.radius*np.cos(th[-1])],
                                 [y[-1], y[-1] + obj.radius*np.sin(th[-1])],
                                 color="k")

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')

        if ax is None: plt.show()


    def animate_results(self, obj, state, dt=0.01):
        """ Animates simulation trajectory """

        # Find axes limits
        min_x = min(state[0,:])
        max_x = max(state[0,:])
        min_y = min(state[1,:])
        max_y = max(state[1,:])
        xrange = max_x - min_x
        yrange = max_y - min_y
        xbuf = 0.1*xrange
        ybuf = 0.1*yrange
        num_pts = state.shape[1]

        # Initialize figures
        fig = plt.figure(1, figsize=(12,8))
        ax = fig.add_subplot(111, aspect="equal")
        plt.title("Throwing Trajectory")
        xdata, ydata = [], []
        line, = ax.plot([], [], "-")
        circ = plt.Circle((0,0), obj.radius, color=obj.color)
        ori_line, = ax.plot([0, obj.radius],[0,0], color="k")

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

            x_ori = [x, x + obj.radius*np.cos(th)]
            y_ori = [y, y + obj.radius*np.sin(th)]
            ori_line.set_data(x_ori, y_ori)

            # Close the figure if this is the last step
            # NOTE: This is not reliable, and will lead to a core dump if
            # you use blit=True in animation.FuncAnimation
            # (even if it does speed up the animation itself)
            # if i == num_pts-1:
            #     plt.close()
            #     plt.pause(0.5)
            #     return tuple()
            return line, circ, ori_line

        pts_skip = int(dt/self.dt)
        frames = np.arange(0, state.shape[1], pts_skip)
        ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=frames,
            blit=False, repeat=False, interval=dt*1000)
        plt.show()
