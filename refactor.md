# Structuring the MVP code to be modular

### Agent (agents/*.py)

Agents are entities that are responsible for manipulating the "real world". Each agents maintains its own environment. In theory, they do not have access to the latent properties of an object (although they may have access to these objects in code, they do not use this information). Each agent will adhere to the following interface:

 * init(blocks): Initialize a world with the given blocks.
 * simulate_action(action, block): Perform the given action in the "real world" and return the observation. An observation is a tuple: `(action, T, end_pose)`. Including the action is required for the filter.
 * simulate_tower()

 We support two types of agents discussed in the following subsections.

#### TeleportAgent (agents/teleport_agent.py)

`TeleportAgent` is an Agent that simply teleports blocks to their desired poses. This is intended for testing/debugging of the planners independently from a robotic platform.

#### PandaAgent (agents/panda_agent.py)

`PandaAgent` is a simulated Panda robot. It instatiates the block world and any actions are implemented by the manipulator. To aid this, a simple pick and place domain is modeled using PDDLStream (see `tamp/design.md`).

### ParticleBelief(Belief) –- Belief Update (filter.py)

Given an observation, update the belief. This is a particle filter for each object. Has a pybullet server. The belief uses a "platform world" to simulate updates. The only objects that exist in the world are a platform and the block of interest.

 * init(obj, num_particles): Initialize a `ParticleDistribution` with N particles uniformly sampled within the extent of a block. 
 * update(observation): Update the particle distribution by filtering on the new observation. Includes a Metropolis-Hastings reampling step to make sure newly sampled particles agree with the previous observations (we don't want to only randomly resample particles). Because of this, a PyBullet simulation is used both for the resampling step and calculating the likelihood of each particle.

### plan_action(belief) -- Planning for curiosity (actions.py)

Given a belief, generate a new action (PDDLStream) to puruse. Choose which
object and what to do with it. This uses the pybullet server in belief to
simulate the outcomes of various actions.
``` 
plan_action(belief, k, exp_type, action_type) -> ActionBase
```

Current supported actions are `PlaceAction` and `PushAction`:
 * `PlaceAction`: Put the object on the edge of a platform and observe how it falls.
 * `PushAction`: Push an object off a platform and observe how it falls.

`exp_type=['random', 'reduce_var']` currently dictates the action selection strategy. `'random'` chooses a random action while `reduce_var` takes into account the current belief of the latent property and chooses an action to reduce its variance. Currently `'reduce_var'` is only supported by `PushAction`.

Also note that the `PandaAgent` currently only supports `PlaceAction`, while `TeleportAgent` only supports `PushAction`.

The place action is currently executed in the `make_platform_world` function by initializing the pose to that specified by the action. `step` does not do anything.
 
### TowerPlanner(Planner) -- Planning for final task (stability.py)

Given a belief a test-time objective, output a "plan" (PDDLStream). Each final
task should be its own class that implements

 * plan(blocks) -> List(Block): Each Block in the returned list has the pose it should take on in the final tower.

-------------------------------------------------------------------------------

## Utility Objects

### ParticleDistribution (block_utils.py)

Represents a distribution by a set of `N` samples and their corresponding weights. This is a `namedtuple` with fields:
 * `particles`: (N x D) numpy array where the domain of the distribution has `D` dimensions.
 * `weights`: (N) numpy array where each entry corresponds to the particle of the same index.


### TODO

 * <s>*Both* Grok PDDLstream spec</s> [DONE]
 * *Together* Write a Planner base class
 * *Together* Refactor filter.py into ParticleBelief [DONE]
 * <s>*Izzy* refactor stability.py into TowerPlanner</s> [DONE]
 * *Mike* refactor actions.py into InfoPlanner [partially DONE as plan_actions]
 
 * Refactor code to create `TeleportAgent`.
 * Implement `PandaAgent`:
    * Create a PDDLStream world in the init function.
    * Translate the results of a PlaceAction to a PDDLStream problem.
    * Translate the results of a TowerPlan plan to a PDDLStream problem.
 * Implement `reduce_var` InfoPlanner strategy for `PlaceAction`
 

 * *Maybe* rewrite run.py to work with new class structure and take PPDLStream
 * *Maybe* Move from namedtuples to np arrays for Position, Rotation, etc


Server > Environment > (World = Particle)