"""
Attempting to make this return a similar plot as in the GAIL paper, Figure 1,
and also to return a table with results.

(c) June 2017 by Daniel Seita
"""

import argparse
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Some matplotilb options I like to use
plt.style.use('seaborn-darkgrid')
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 12
mew = 5
error_region_alpha = 0.25

task_to_name = {'cartpole': 'CartPole-v0',
                'mountaincar': 'MountainCar-v0'}
task_to_random = {'cartpole': 20.08,
                  'mountaincar': -200.0}
colors = {'red', 'blue', 'yellow', 'black'}


def main():
    """ Here, `resultfile` should be an `.h5` file with all results from a
    category of trials, e.g. classic imitation runs. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('result_h5file', type=str)
    args = parser.parse_args()

    with pd.HDFStore(args.result_h5file, 'r') as f:
        df = f['results']
        unique_tasks = np.unique(df['task'].values)
        unique_algs = np.unique(df['alg'].values)
        unique_numtrajs = np.unique(df['num_trajs'].values)
        assert len(unique_algs) == len(colors)
        
        for task in unique_tasks:
            print("\nPlotting for task/env = {}".format(task_to_name[task]))
            task_df = df[df.task == task]
            
            # Handle the expert plots. They should be repeated across rows.
            # Right now we have 10 (well, it's actually 13 ...) because we
            # sampled 10 expert trajectories in phase 0. An alternative to doing
            # this is to sample 50 expert trajectories in phase 0. However, note
            # that we'll only consider the first 10 of those for any run/trial.

            experts = task_df['ex_traj_returns'].values
            for arr in experts[1:]:
                assert np.sum(np.abs(arr - experts[0])) <= 1e-6
            print("expert mean: {} and std: {} and length: {}".format(
                experts[0].mean(), experts[0].std(), len(experts[0])))
            fig = plt.figure(figsize=(10,8))
            plt.axhline(experts[0].mean(), color='gray', label='Expert')
            plt.axhline(task_to_random[task], color='lightblue', label='Random')

            for (c,alg) in zip(colors,unique_algs):
                taskalg_df = task_df[task_df.alg == alg]
                rew_mean = []
                rew_std = []

                for numtraj in unique_numtrajs:
                    # I.e. a dataset size ("numtraj") of 1, 4, 7, or 10.
                    taskalgnum_df = taskalg_df[taskalg_df.num_trajs == numtraj]

                    # ----------------------------------------------------------
                    # These are lists of numpy arrays of length equal to the
                    # number of runs we ran, which is 7 by default. The numpy
                    # arrays should be of length equal to the number of
                    # trajectories we ran for evaluation, which *should* be 50,
                    # but we're getting 52 or 53 for some reason ?!?!?  Also, I
                    # assume we'll concatenate everything? Yeah, the appendix of
                    # the GAIL paper implies this by saying that statistics are
                    # "further computed" from 7 initializations.
                    # ----------------------------------------------------------

                    alg_traj_lengths = taskalgnum_df['alg_traj_lengths'].values
                    alg_traj_returns = taskalgnum_df['alg_traj_returns'].values
                    concat_returns = np.concatenate([x for x in alg_traj_returns])
                    rew_mean.append(concat_returns.mean())
                    rew_std.append(concat_returns.std())

                rew_mean = np.array(rew_mean)
                rew_std = np.array(rew_std)
                plt.plot(unique_numtrajs, rew_mean, '-x', lw=lw, color=c,
                        markersize=ms, mew=mew, label=alg)
                plt.fill_between(unique_numtrajs, 
                                 rew_mean-rew_std,
                                 rew_mean+rew_std, 
                                 alpha=error_region_alpha, 
                                 facecolor=c)

            plt.title(task_to_name[task], fontsize=title_size)
            plt.xlabel("Number of Trajectories", fontsize=ysize)
            plt.ylabel("Scores", fontsize=xsize)
            plt.legend(loc='lower right', prop={'size':legend_size})
            plt.tick_params(axis='x', labelsize=tick_size)
            plt.tick_params(axis='y', labelsize=tick_size)
            plt.tight_layout()
            plt.savefig(FIGDIR+task_to_name[task]+'.png')


if __name__ == '__main__':
    main()
