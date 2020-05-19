# Structuring the MVP code to be modular

### Agent (run.py)

 * belief
 * planners

### ParticleBelief(Belief) –- Belief Update (filter.py)

Given an observation, update the belief. This is a particle filter for
each object. Has a pybullet server.

 * init(list_of_obj)
 * reset()
 * update(observation_traj)
 * get_obj(name)

### InfoPlanner(Planner) -- Planning for curiosity (actions.py)

Given a belief, generate a new action (PDDLStream) to puruse. Choose which
object and what to do with it. This uses the pybullet server in belief to
simulate the outcomes of various actions

 * plan(belief) -> PDDLStream

### TowerPlanner(Planner) -- Planning for final task (stability.py)

Given a belief a test-time objective, output a "plan" (PDDLStream). Each final
task should be its own class that implements

 * plan(belief) -> PDDLStream

-------------------------------------------------------------------------------

### TODO

 * *Both* Grok PDDLstream spec
 * *Together* Write a Planner base class
 * *Together* Refactor filter.py into ParticleBelief
 * *Izzy* refactor stability.py into TowerPlanner
 * *Mike* refactor actions.py into InfoPlanner
 * *Maybe* rewrite run.py to work with new class structure and take PPDLStream
 * *Maybe* Move from namedtuples to np arrays for Position, Rotation, etc


Server > Environment > (World = Particle)