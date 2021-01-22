"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
from block_utils import *
from filter_utils import *
from base_class import PlannerBase
import numpy as np
from copy import copy, deepcopy
from itertools import permutations, combinations_with_replacement


# TODO(izzy): right now I'm assuming sequential vertical contacts. This get's a
# lot more tricky if we want to place blocks adjacent to eachother

class TowerPlanner(PlannerBase):
    def __init__(self, stability_mode='angle', plan_mode='confidence'):
        """
        Keyword Arguments:
            stability_mode {str} -- set the mode for checking stability
                'angle': check the angle from the pivot to the top COM
                'contains': check whether the top COM is in the bottom geom
            plan_mode {str} -- set the mode for the tower planning objective
                'expectation': build the tallest tower in expectation
                'confidence':  bulid the tallest tower with a certain stability
        """
        self.stability_mode = stability_mode
        self.plan_mode = plan_mode

        self.angle_thresh = 3    # degrees
        self.confidence_thresh = 0.9 # how likely is the tallest tower stable


    def tower_is_stable(self, tower):
        """ Check that the tower is stable

        NOTE: This function expects blocks in the tower to have zero rotation

        Arguments:
            blocks {List(Object)} -- the tower from bottom to top

        Returns:
            bool -- Whether or not the tower is stable
        """

        top_group = tower[-1]
        # iterate down the tower, checking stability along the way. Is the group of
        # blocks above the current block stable on the current block?
        # we don't check the top block because there's nothing on top of it
        for block in reversed(tower[:-1]):
            # check stability
            if not self.pair_is_stable(block, top_group):
                return False

            # add the block to the group
            top_group = group_blocks(block, top_group)

        return True # we've verified the whole tower is stable

    def tower_is_constructable(self, blocks):
        """ Check that each block can be placed on the tower from bottom to top
        without knocking the tower over

        NOTE: This function expects blocks in the tower to have zero rotation

        Arguments:
            blocks {List(Object)} -- the tower from bottom to top

        Returns:
            bool -- Whether or not the tower can be built stably
        """
        for i in range(1, len(blocks)+1):
            # check that each pair of blocks is stably individually
            subtower = blocks[:i]
            if not self.tower_is_stable(subtower): return False

        return True

    def tower_is_pairwise_stable(self, blocks):
        """ Check that each pair of blocks is stable wrt the one below it.

        NOTE: This function expects blocks in the tower to have zero rotation

        Arguments:
            blocks {List(Object)} -- the tower from bottom to top

        Returns:
            bool -- Whether or not the tower can be built stably
        """
        for i in range(len(blocks) - 1):
            # check that each pair of blocks is stably individually
            top = blocks[i+1]
            bottom = blocks[i]
            if not self.pair_is_stable(bottom, top): return False

        return True

    def pair_is_stable(self, bottom, top):
        """ Return True if the top object is stable on the bottom object

        Arguments:
            bottom {Object} -- [description]
            top {Object} -- [description]
        """

        if self.stability_mode == 'angle':
            # position of the top block relative to the bottom COG
            top_rel_pos = np.array(top.pose.pos) - np.array(bottom.pose.pos)
            # position of the top COM relative to the bottom COG
            top_rel_com = top_rel_pos + np.array(top.com)
            # calculate the four pivot edges
            pivot_x_p = min(bottom.dimensions.x/2, top_rel_pos[0] + top.dimensions.x/2)
            pivot_x_n = max(-bottom.dimensions.x/2, top_rel_pos[0] - top.dimensions.x/2)
            pivot_y_p = min(bottom.dimensions.y/2, top_rel_pos[1] + top.dimensions.y/2)
            pivot_y_n = max(-bottom.dimensions.y/2, top_rel_pos[1] - top.dimensions.y/2)
            # get the height of the COM above the bottom block top surface
            top_com_dist_z_to_bottom_surface = top_rel_com[2] - bottom.dimensions.z/2
            # get the distance of the COM to the four pivot edges. these are positive
            # if the COM is inside the pivot
            top_com_dist_x_p = pivot_x_p - top_rel_com[0]
            top_com_dist_x_n = top_rel_com[0] - pivot_x_n
            top_com_dist_y_p = pivot_y_p - top_rel_com[1]
            top_com_dist_y_n = top_rel_com[1] - pivot_y_n
            # calculate the angles from the pivots to the COM
            y = top_com_dist_z_to_bottom_surface
            xs = np.array([top_com_dist_x_p,
                           top_com_dist_x_n,
                           top_com_dist_y_p,
                           top_com_dist_y_n])
            angles = np.degrees(np.arctan2(y, xs))
            # and check stability
            return (angles < 90 - self.angle_thresh).all()
        else:
            # Check if the COM of the top object is within the dimensions of the bottom
            # object. We assume that the two objects are in static planar contact in z,
            # and that the COM of the top object must lie within the object
            top_rel_pos = np.array(top.pose.pos) - np.array(bottom.pose.pos)
            top_rel_com = top_rel_pos + top.com
            return (np.abs(top_rel_com)*2 - bottom.dimensions <= 0)[:2].all()

    def tower_is_containment_stable(self, tower):
        """ A distractor stability function. Returns true if each block is completely 
            within the block below it. 
        """
        for i in range(len(tower) - 1):
            # check that each pair of blocks is stably individually
            top = tower[i+1]
            bottom = tower[i]
            
            if (top.pose.pos[0] + top.dimensions.x/2.) > (bottom.pose.pos[0] + bottom.dimensions.x/2.):
                return False
            if (top.pose.pos[0] - top.dimensions.x/2.) < (bottom.pose.pos[0] - bottom.dimensions.x/2.):
                return False
            if (top.pose.pos[1] + top.dimensions.y/2.) > (bottom.pose.pos[1] + bottom.dimensions.y/2.):
                return False
            if (top.pose.pos[1] - top.dimensions.y/2.) < (bottom.pose.pos[1] - bottom.dimensions.y/2.):
                return False

        return True


    def tower_is_cog_stable(self, tower):
        """ Return True is the tower would be stable if the CoM==COG
        """
        cog_tower = [deepcopy(block) for block in tower]
        for block in cog_tower:
            block.com = Position(0., 0., 0.)
        return self.tower_is_stable(cog_tower)

    def calc_expected_height(self, tower, num_samples=100):
        """ Finds the expected height of a tower

        If we are uncertain about the center of mass of blocks, then for any
        tower there is some probability that it will collapse. This function
        finds the height of a tower times the probability that it is stable.

        NOTE: This funcion modifies the com field of the blocks in tower.
        NOTE: This function expects blocks in the tower to have zero rotation

        Arguments:
            tower {List(Object)} -- the objects in the tower

        Keyword Arguments:
            num_samples {number} -- how many times to sample COM configurations
                (default:self,  {100})

        Returns:
            [float] -- expected height of the tower
        """
        # sample a bunch of COMs for each block in the tower
        com_samples = {block.name:
            sample_particle_distribution(block.com_filter, num_samples)
            for block in tower}
        stable_count = 0

        # for each possible COM sample, check if such a tower would be stable
        for i in range(num_samples):
            for block in tower:
                block.com = com_samples[block.name][i]

            stable_count += self.tower_is_stable(tower)\
                * self.tower_is_constructible(tower)

        height = np.sum([block.dimensions.z for block in tower])
        p_stable = stable_count / float(num_samples)
        return height, p_stable

    def plan(self, blocks, num_samples=100):
        """ Finds the tallest tower in expectation given uncertainy over COM

        Arguments:
            blocks {List(Object)} -- the blocks in the tower

        Keyword Arguments:
            num_samples {number} -- how many times to sample COM configurations
                (default:self,  {100})

        Returns:
            {List(Object)} -- The objects in the tallest tower (from the bottom up) with
                orientations set
        """
        n = len(blocks)
        max_height = 0
        max_tower = []

        # for each ordering of blocks
        for tower in permutations(blocks):
            # for each combination of rotations
            for block_orientations in combinations_with_replacement(rotation_group(), r=n):
                # set the orientation of each block in the tower
                for block, orn in zip(tower, block_orientations):
                    block.pose = Pose(ZERO_POS, Quaternion(*orn.as_quat()))
                # unrotate the blocks and calculate their poses in the tower
                stacked_tower = self.set_stack_poses(tower)
                # simulate_tower(stacked_tower, vis=True, T=25)
                # and check the expected height of this particular tower
                height, p_stable = self.calc_expected_height(stacked_tower, num_samples=num_samples)

                # save the tallest tower
                if self.plan_mode == 'confidence':
                    if height > max_height and p_stable > self.confidence_thresh:
                        max_tower = stacked_tower
                        max_height = height
                else:
                    expected_height = height * p_stable
                    if expected_height > max_height:
                        max_tower = stacked_tower
                        max_height = height

        return max_tower

    def set_stack_poses(self, blocks):
        """ Find the pose of each block if we were to stack the blocks

        Stacks the blocks in the given order (bottom up). Aligns blocks such that
        the center of mass (or mean of the estimated center of mass given a
        com_filter) are all colinear

        NOTE: if the blocks have a rotation, get_rotated_block will be applied to
        each, so the returned blocks have zero rotation

        Arguments:
            blocks {List(Object)} -- the list of blocks in the tower

        Returns:
            blocks {List(Object)} -- the list of blocks in the tower
        """
        # rotate all the blocks (COM, dimensions) by their defined self, rotations
        blocks = [get_rotated_block(block) for block in blocks]
        prev_z = 0
        for block in blocks:
            pos = np.zeros(3)
            # set the x,y position of the block
            if block.com_filter is not None:
                pos[:2] = -get_mean(block.com_filter)[:2]
            else:
                pos[:2] = -np.array(block.com)[:2]
            # set the relative z position of the block
            pos[2] = prev_z + block.dimensions.z/2
            # and update the block with the desired pose
            block.pose = Pose(Position(*pos), ZERO_ROT)
            # save the height of the top of the block
            prev_z += block.dimensions.z

        return blocks

