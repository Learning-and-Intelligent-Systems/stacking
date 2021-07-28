# stacking

This repo has code for the paper [Active Learning of Abstract Plan Feasibility](https://arxiv.org/abs/2107.00683). In it we perform experiments in a block stacking domain where the evaluation tasks include constructing the tallest tower, longest overhang, etc. using a robot manipulator. This repo includes:

* A simulation-based particle filter to estimate the center of mass of blocks.
* An active learning strategy to learn a model for abstract plan feasibility with a information-theoretic approach to improve data sampling efficiency.
* (IN PROGRESS) A learning approach to estimate task-specific block properties using a learned latent representation.

All robot experiments (simulation and real-world) are conducted using a Franka Emika Panda robot.

For a more detailed overview of the repository and its folders, refer to [the overview README](./doc/overview.md).

---

## Installation
See the full installation steps in [the installation README](./doc/installation.md).

---

## Hardware Setup
For information on our hardware configuration, refer to [our hardware setup README](./doc/hardware.md).

For specific information on the blocks used for our experiments, refer to [our blocks README](./doc/blocks.md). This README also describes how to draw initial "home" block poses on the table.

---

## Running Examples
See the full execution guide in [the execution README](./doc/execution.md).

---

## Troubleshooting

Finally, refer to our [troubleshooting README](./doc/troubleshooting.md), contact us, or raise an issue if you are facing any problems.
