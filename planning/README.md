plan.py is the script that has the plan_mcts() function in it. You can run this script (with command line args) to compare different c values and also to just perform multiple runs under the same parameters. When you run searches with this file, figures will be output (to --exp-path/figures) that show:
1. a histogram of the distribution of tower heights in the search tree over time
2. the highest UCT value in the tree over time
3. the highest expected value in the tree over time
4. the number of blocks in the tower with the most blocks in the tree over time

run.py is the top level script for running planning (either sequential or total/random planning). When run from here a lot of pickle files will be generated.
You can use plot_comparison.py to generate plots from these pickle files.
For sequential and total planning you can plot:
1. the median tower height found over training time

For sequential planning you can plot:
1. the number of nodes in the search tree over training towers
2. the number of towers with the max number of blocks in them in the search tree over training towers
3. the median value of the nodes in the tower broken up by the number of blocks in the tower over time (NOT training towers)

