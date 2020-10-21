The following datasets can be used for training:

`random_blocks_(x40000)_5blocks_uniform_mass.pkl`: Contains 10k examples/tower size (ranging from 2-5 blocks). This can be reduced in the script that loads the data.

`random_blocks_(x40000)_5blocks_uniform_mass_aug_4.pkl`: Uses every 4th tower from the original dataset: 10k unique towers/size (40k augmented).
`random_blocks_(x40000)_5blocks_uniform_mass_aug_8.pkl`: Uses every 8th tower from the original dataset: 5k unique towers/size (20k augmented).
`random_blocks_(x40000)_5blocks_uniform_mass_aug_16.pkl`: Uses every 16th tower from the original dataset: 2.5k unique towers/size (10k augmented).
`random_blocks_(x40000)_5blocks_uniform_mass_aug_32.pkl`: Uses every 32nd tower from the original dataset: 1.25k unique towers/size (5k augmented).
`random_blocks_(x40000)_5blocks_uniform_mass_aug_64.pkl`: Uses every 64th tower from the original dataset: 625 unique towers/size (2.5kk augmented).

The following is the test set:
`random_blocks_(x2000)_5blocks_uniform_mass.pkl`: Contains 2k examples/tower size.
