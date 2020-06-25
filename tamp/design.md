# Notes on Demain Design

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