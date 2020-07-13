# Notes on Demain Design

## Platform Representation

We use a platform object to aid in experimental setup. In PDDLStream, we simply represent this as another `Block` but do not assign it the `Graspable` predicate. This ensures we don't try to move the platform but inherits other useful properties of being a block (e.g., the ability to sample a position on top of it using a relative pose).

## Grasping Considerations

### Grasping for Platform Placements 

It is useful for the platform to be tall as this increases the likelihood that a sampled grasp and pose won't cause collisions with the table. If the planner appears to be taking a while to place a block on the platform, it is worth manually evaluating the streams (`agents/panda_agent.py:test_placement_on_platform,test_table_pose_ik`) to make sure valid IK solutions can be found. 

In the future it may help to figure out how to better sample poses that are useful for regrasping (sampling random poses on top of the table is not efficient and can easily lead to collisions as the gripper is near the table).

It is also helpful for regrasping attempts to make the long end of the block sufficiently large (> 8cm) so that a side-grasp from this side does not result in a table collision.

Since we have a lot of grasps, it is useful to have a high `search_sample_ratio` so that we spend more time trying to sample placements that work (as opposed to coming up with more complex plans). This makes it hard to find plans where blocks need to be moved out of each other's way. We might need a better solution to handle this (i.e., create more efficient streams). One way to do this would be to be more careful about grasp generation. We could make sure we never try to grasp in orientations that do not fit in the gripper.

### Grasping for Moving Fallen Blocks

Once a block falls off the platform, its position is often too close to the platform to be able to pick up. Even if a grasp is available, it's also likely later aspects of the plan will fail because it collides with the platform when initially moving the block.

Some ideas:
- Make platform more of a table.
- Add 45 degree angle grasps.
- Change approach to first move block away from table.
- Add pushing action.

Current Plan:
- First make the platform more of a table so the block can fall under without being stuck against a wall.
- Next define grasps that pick up the block at an angle.
- Make the angle low enough and combine with a larger approach which moves the block away from the table.

## Relative vs. Aboslute Poses

Absolute poses are useful for solving IK as we need to know the location of the object in the robot frame. However, when expressing the tower to build, we only care about the relative poses between blocks - the base of the tower can exist anywhere on the table.

`RelPose(?o1 ?o2 ?rp)` captures the desired relative pose between two blocks' center of geometries. This will be provided by the external belief-space planner. One way to incorporate this information into the planner would be:
- Specify goals using `On(?o1, ?o2, ?rp)` predicates. 
- Initialize the state with `RelPose` predicates from the external planner.
- In the stream `sample-pose-block`, use the provided `RelPose` to sample the global pose of the top block (which can then be checked for IK).

Given the second two bullet points above, it is unnecessary to specify the `RelPose` in the goal `On` predicate. Since the stream will only sample valid relative poses for block combinations, `On(?o1 ?o2)` is sufficient to express the desired goals.

## Collision Checking

When collision checking, we need to check against fixed objects whose positions never change (such as the table), as well as object's whose positions may change throughout the plan (blocks). There are two approaches we can use to do this.

### Using Tests

Tests are streams which output a Boolean value. One way to check if a trajectory is collision free would be to define a test that checks whether a trajectory collides with an object:
```
(:stream check-moving-collision
    :inputs (?t ?o ?p)
    :domain (and (Pose ?o ?p) (Block ?o) (Traj ?t))
    :certified (TrajSafe ?t ?o ?p)
)
```
A precondition for any action that involves a trajectory would need to ensure that for that trajectory, there are no collisions at the current poses of the objects:
```
(:derived (UnsafeTraj ?t)
    (exists (?o3 ?p3) (and (AtPose ?o3 ?p3) (Block ?o3)
                           (not (TrajSafe ?t ?o3 ?p3)))
    )
)
```
Unfortunately, I was never able to get this method working. The stream was never actually called during planning (although the preconditions must have been true because it found a plan).

### Using Fluent Streams

Another method is to use fluent information when evaluating a stream. This can be done by adding a fluents field to the stream definition:
```
(:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (and (FreeMotion ?q1 ?t ?q2))
)
```
When the stream function is called, it will be passed a fluents list as an argument with all fluents that are currently true with the given predicate.

Note that fluent predicates can't be used when they certify predicates that are also inputs to other streams. The implication for us is that we can't do fluent collision checking in the IK stream (instead trajectory generation should be moved to the plan-motion streams). I still need to update the IK-stream to fix this.

This is the method we use.