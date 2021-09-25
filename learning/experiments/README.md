# Training Phase

To train a latent model, see `learning/experiments/train_base_model.sh`.

# Fitting Phase

## Data Collection Method: Task-Informed vs. Task-Uninformed

Fitting can take place in a task-informed or uninformed way. The task uninformed method is the method proposed in our work and makes use of information gain approximations to gather data. Task-informed uses a baseline method to guide exploration.

Both the fitting methods can use a task to collect data.

## Fitting Method: Variational Inference vs. Particle Filtering