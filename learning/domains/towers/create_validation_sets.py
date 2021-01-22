from block_utils import Object
from learning.domains.towers.generate_tower_training_data import build_tower, vectorize, sample_with_replacement
from tower_planner import TowerPlanner
import numpy as np
import pickle


def is_robust(orig_tower, n_attempts=10, noise=0.001):
    """ Perturb each block in the tower by 1mm multiple times and make sure the label does not change. """
    tp = TowerPlanner(stability_mode='contains')
    robust = True
    tower_vec = np.array([orig_tower[bx].vectorize() for bx in range(0, len(orig_tower))])
    label = tp.tower_is_constructable(orig_tower)
    for _ in range(n_attempts):
        tower = tower_vec.copy()
        tower[:, 7:9] += np.random.randn(2*tower.shape[0]).reshape(tower.shape[0], 2)*noise

        block_tower = [Object.from_vector(tower[kx, :]) for kx in range(tower.shape[0])]

        if tp.tower_is_constructable(block_tower) != label:
            robust = False

    return robust


def main(vis_tower=False):
    num_towers = 500

    pw_stable = True
    # create a vector of stability labels where half are unstable and half are stable
    stability_labels = np.zeros(num_towers*2, dtype=int)
    stability_labels[500:] = 1.

    dataset = {}
    for num_blocks in range(3, 6):
        vectorized_towers = []

        for constructable in [False, True]:
            count = 0
            while count < num_towers:
                # print the information about the tower we are about to generate
                stability_type = "con" if constructable else "uncon"
                stability_type += "/pw_stable" if pw_stable else "/pw_unstable"
                print(f'{count}/{num_towers}\t{stability_type} {num_blocks}-block tower')

                # generate random blocks. Use the block set if specified. otherwise
                # generate new blocks from scratch. Save the block names if using blocks
                # from the block set
                blocks = [Object.random(f'obj_{ix}') for ix in range(num_blocks)]
                cog_stable = np.random.choice([True, False])
                tower = build_tower(blocks, 
                                    constructable=constructable, 
                                    pairwise_stable=pw_stable, 
                                    cog_stable=cog_stable)
                
                if tower is None:
                    continue
                if not is_robust(tower):
                    print('Not Robust')
                    continue


                if vis_tower:
                    w = World(tower)
                    env = Environment([w], vis_sim=True, vis_frames=True)
                    print(stability_type)
                    input()
                    for tx in range(240):
                        env.step(vis_frames=False)
                        time.sleep(1/240.)
                    env.disconnect()

                count += 1
                # append the tower to the list
                vectorized_towers.append(vectorize(tower))
            
        data = {
            'towers': np.array(vectorized_towers),
            'labels': stability_labels,
        }


        dataset[f'{num_blocks}block'] = data

    # save the generate data
    filename = 'learning/data/validation_towers_robust.pkl'
    print('Saving to', filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
