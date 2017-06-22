import argparse
import json
import h5py
import numpy as np
import yaml
import sys
import os, os.path, shutil
from policyopt import util
import subprocess, tempfile, datetime


def load_trained_policy_and_mdp(env_name, policy_state_str):
    """ Creates the specialized MDP and policy objects needed to sample expert
    trajectories for a given environment.

    Returns:
        mdp: An instance of `RLGymMDP`, similar to a real gym env except with
            customized obs/action spaces and an internal `RLGyMSim` object.
        policy: The agent's policy, encoded as either rl.GaussianPolicy for
            continuous actions, or rl.GibbsPolicy for discrete actions.
        train_args: A dictionary of arguments (like argparse dicts) based on the
            trained policy's TRPO run.
    """
    import gym
    import policyopt
    from policyopt import nn, rl
    from environments import rlgymenv

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(policy_state_str)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])

    # Initialize the MDP
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    print 'MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size)

    # Initialize the policy
    nn.reset_global_scope()
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

    # Load the policy parameters
    policy.load_h5(policy_file, policy_key)

    return mdp, policy, train_args


def gen_taskname2outfile(spec, assert_not_exists=False):
    '''
    (JHo) Generate dataset filenames for each task. Phase 0 (sampling) writes to
    these files, phase 1 (training) reads from them.
    '''
    taskname2outfile = {}
    trajdir = os.path.join(spec['options']['storagedir'], spec['options']['traj_subdir'])
    util.mkdir_p(trajdir)
    for task in spec['tasks']:
        assert task['name'] not in taskname2outfile
        fname = os.path.join(trajdir, 'trajs_{}.h5'.format(task['name']))
        if assert_not_exists:
            assert not os.path.exists(fname), 'Traj destination {} already exists'.format(fname)
        taskname2outfile[task['name']] = fname
    return taskname2outfile


def exec_saved_policy(env_name, policystr, num_trajs, deterministic, max_traj_len=None):
    """ For given env_name (e.g. CartPole-v0)  with an expert policy, call
    `mdp.sim_mp` to get states and actions info for the sampled trajectories.
    
    The simulation's actually defined in `policyopt/__init__.py`, because the
    mdp (RLGymMDP) is a subclass of the MDP there. It mostly relies on a
    `sim_single` method which simulates one rollout in very readable code.
    
    Returns
    -------
    trajbatch: [TrajBatch]
        A customized class encoding information about the trajectories, e.g. it
        can handle varying lengths.
    policy: [rl.Policy]
        The agent's policy, encoded as either rl.GaussianPolicy for continuous
        actions, or rl.GibbsPolicy for discrete actions.
    mdp: [RLGymMDP]
        Similar to a real gym env except with customized obs/action spaces and
        an `RLGyMSim` object.
    """
    import policyopt
    from policyopt import SimConfig, rl, util, nn, tqdm
    from environments import rlgymenv
    import gym

    # Load MDP and policy
    mdp, policy, _ = load_trained_policy_and_mdp(env_name, policystr)
    max_traj_len = min(mdp.env_spec.timestep_limit, max_traj_len) if max_traj_len is not None else mdp.env_spec.timestep_limit
    print 'Sampling {} trajs (max len {}) from policy {} in {}'.format(num_trajs, max_traj_len, policystr, env_name)

    # Sample trajs
    trajbatch = mdp.sim_mp(
        policy_fn=lambda obs_B_Do: policy.sample_actions(obs_B_Do, deterministic),
        obsfeat_fn=lambda obs:obs,
        cfg=policyopt.SimConfig(
            min_num_trajs=num_trajs,
            min_total_sa=-1,
            batch_size=None,
            max_traj_len=max_traj_len))

    return trajbatch, policy, mdp


def eval_snapshot(env_name, checkptfile, snapshot_idx, num_trajs, deterministic):
    """ Called during evaluation stage, prints results on screen and returns
    data which we save in a results `.h5` file. """
    policystr = '{}/snapshots/iter{:07d}'.format(checkptfile, snapshot_idx)
    trajbatch, _, _ = exec_saved_policy(
        env_name,
        policystr,
        num_trajs,
        deterministic=deterministic,
        max_traj_len=None)
    returns = trajbatch.r.padded(fill=0.).sum(axis=1)
    lengths = np.array([len(traj) for traj in trajbatch])
    util.header('{} gets return {} +/- {}'.format(policystr, returns.mean(), returns.std()))
    return returns, lengths


def phase0_sampletrajs(spec, specfilename):
    """ The first phase, sampling expert trajectories from TRPO. 
    
    This *can* be done sequentially on one computer, no need to worry. This
    *will* save the .h5 files according to `storagedir` in the specs, so
    manually remove if needed. 
    
    This will sample `full_dataset_num_trajs` expert trajectories. I think it
    might be better to have that value be perhaps 50, since then I can use those
    values directly when plotting the expert performance alongside the
    algorithms, to be consistent in getting 50 samples.
    
    Just note that sampling more than 10 trajectories (or whatever our limit is)
    will **not** change the actual dataset, i.e. if we need 10 out of 20
    trajectories, the `load_datasets` method will always load the first 10, and
    not randomly pick 10 out of the 20.
    """
    util.header('=== Phase 0: Sampling trajs from expert policies ===')
    num_trajs = spec['training']['full_dataset_num_trajs']
    util.header('Sampling {} trajectories'.format(num_trajs))

    # Make filenames and check if they're valid first
    taskname2outfile = gen_taskname2outfile(spec, assert_not_exists=True)

    # Sample trajs for each task
    for task in spec['tasks']:
        # Execute the policy
        trajbatch, policy, _ = exec_saved_policy(
            task['env'], task['policy'], num_trajs,
            deterministic=spec['training']['deterministic_expert'],
            max_traj_len=None)
        
        # Quick evaluation
        returns = trajbatch.r.padded(fill=0.).sum(axis=1)
        avgr = trajbatch.r.stacked.mean()
        lengths = np.array([len(traj) for traj in trajbatch])
        ent = policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean()
        print 'returns.shape: {}'.format(returns.shape)
        print 'ret: {} +/- {}'.format(returns.mean(), returns.std())
        print 'avgr: {}'.format(avgr)
        print 'len: {} +/- {}'.format(lengths.mean(), lengths.std())
        print 'ent: {}'.format(ent)

        # Save the trajs to a file. Pad in case uneven lengths, but typically
        # the experts last the full duration so the lengths will be equivalent.
        with h5py.File(taskname2outfile[task['name']], 'w') as f:
            def write(dsetname, a):
                f.create_dataset(dsetname, data=a, compression='gzip', compression_opts=9)
            # Right-padded trajectory data using custom RaggedArray class.
            write('obs_B_T_Do', trajbatch.obs.padded(fill=0.))
            write('a_B_T_Da', trajbatch.a.padded(fill=0.))
            write('r_B_T', trajbatch.r.padded(fill=0.))
            # Trajectory lengths
            write('len_B', np.array([len(traj) for traj in trajbatch], dtype=np.int32))
            # # Also save args to this script
            # argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
            # f.attrs['args'] = argstr
        util.header('Wrote {}'.format(taskname2outfile[task['name']]))


def phase1_train(spec, specfilename):
    """ In the normal code, this rounds up a long list of commands of the form
    `python (script name) (arguments)` which can be run on a cluster.

    It's really cool how this works. The `cmd_templates` list turns into a bunch
    of python script calls, except it has string formatting to allow the
    arguments to fill them in. A much better way than writing a long bash
    script! (Actually, to *get* a bash script, just write these one by one to a
    file and then I think running the file is OK.)

    I modified this to run sequentially.
    """
    util.header('=== Phase 1: training ===')

    # Generate array job that trains (1) all algorithms over (2) all tasks, for
    # (3) all dataset sizes, so yes it's three loops.
    taskname2dset = gen_taskname2outfile(spec)

    # Make checkpoint dir. All outputs go here
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    util.mkdir_p(checkptdir)
    # Make sure checkpoint dir is empty
    assert not os.listdir(checkptdir), 'Checkpoint directory {} is not empty!'.format(checkptdir)

    # Assemble the commands to run on the cluster
    cmd_templates, outputfilenames, argdicts = [], [], []
    for alg in spec['training']['algorithms']:
        for task in spec['tasks']:
            for num_trajs in spec['training']['dataset_num_trajs']:
                assert num_trajs <= spec['training']['full_dataset_num_trajs']
                for run in range(spec['training']['runs']):
                    # A string identifier. Used in filenames for this run
                    strid = 'alg={},task={},num_trajs={},run={}'.format(alg['name'], task['name'], num_trajs, run)
                    cmd_templates.append(alg['cmd'].replace('\n', ' ').strip())
                    outputfilenames.append(strid + '.txt')
                    argdicts.append({
                        'env': task['env'],
                        'dataset': taskname2dset[task['name']],
                        'num_trajs': num_trajs,
                        'cuts_off_on_success': int(task['cuts_off_on_success']),
                        'data_subsamp_freq': task['data_subsamp_freq'],
                        'out': os.path.join(checkptdir, strid + '.h5'),
                    })

    # (New code from Daniel) Put commands in a list and run them sequentially.
    all_commands = [x.format(**y) for (x,y) in zip(cmd_templates,argdicts)]
    print("Total number of commands to run: {}.".format(len(all_commands)))
    for command in all_commands:
        subprocess.call(command.split(" "))


def phase2_eval(spec, specfilename):
    """ Evaluates results from a given set of experiments, specified in the
    .yaml files.

    This is all saved into `results.h5`, which one should then probably
    rename and move somewhere else to save. Also, you'll need to figure out how
    to plot from those files.
    """
    util.header('=== Phase 2: evaluating trained models ===')
    import pandas as pd

    taskname2dset = gen_taskname2outfile(spec)

    # This is where model logs are stored.  We will also store the evaluation
    # here, from `results_filename`.
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    print 'Evaluating results in {}'.format(checkptdir)

    results_full_path = os.path.join(checkptdir, spec['options']['results_filename'])
    print 'Will store results in {}'.format(results_full_path)
    if os.path.exists(results_full_path):
        raise RuntimeError('Results file {} already exists'.format(results_full_path))

    # First, pre-determine which evaluations we have to do
    evals_to_do = []
    nonexistent_checkptfiles = []
    for task in spec['tasks']:
        # See how well the algorithms did...
        for alg in spec['training']['algorithms']:
            # ...on various dataset sizes
            for num_trajs in spec['training']['dataset_num_trajs']:
                # for each rerun, for mean / error bars later
                for run in range(spec['training']['runs']):
                    # Make sure the checkpoint file exists (maybe PBS dropped some jobs)
                    strid = 'alg={},task={},num_trajs={},run={}'.format(alg['name'], task['name'], num_trajs, run)
                    checkptfile = os.path.join(checkptdir, strid + '.h5')
                    if not os.path.exists(checkptfile):
                        nonexistent_checkptfiles.append(checkptfile)
                    evals_to_do.append((task, alg, num_trajs, run, checkptfile))

    if nonexistent_checkptfiles:
        print 'Cannot find checkpoint files:\n', '\n'.join(nonexistent_checkptfiles)
        raise RuntimeError

    # Walk through all saved checkpoints
    collected_results = []
    for i_eval, (task, alg, num_trajs, run, checkptfile) in enumerate(evals_to_do):
        util.header('Evaluating run {}/{}: alg={},task={},num_trajs={},run={}'.format(
            i_eval+1, len(evals_to_do), alg['name'], task['name'], num_trajs, run))

        # Load the task's traj dataset to see how well the expert does
        with h5py.File(taskname2dset[task['name']], 'r') as trajf:
            # Expert's true return and traj lengths
            ex_traj_returns = trajf['r_B_T'][...].sum(axis=1)
            ex_traj_lengths = trajf['len_B'][...]

        # Load the checkpoint file
        with pd.HDFStore(checkptfile, 'r') as f:
            log_df = f['log']
            log_df.set_index('iter', inplace=True)

            # Evaluate true return for the learned policy
            if alg['name'] == 'bclone':
                # Pick the policy with the best validation accuracy
                best_snapshot_idx = log_df['valacc'].argmax()
                alg_traj_returns, alg_traj_lengths = eval_snapshot(
                    task['env'], checkptfile, best_snapshot_idx,
                    spec['options']['eval_num_trajs'], deterministic=True)

            elif any(alg['name'].startswith(s) for s in ('ga', 'fem', 'simplex')):
                # Evaluate the last saved snapshot
                snapshot_names = f.root.snapshots._v_children.keys()
                assert all(name.startswith('iter') for name in snapshot_names)
                snapshot_inds = sorted([int(name[len('iter'):]) for name in snapshot_names])
                best_snapshot_idx = snapshot_inds[-1]
                alg_traj_returns, alg_traj_lengths = eval_snapshot(
                    task['env'], checkptfile, best_snapshot_idx,
                    spec['options']['eval_num_trajs'], deterministic=True)

            else:
                raise NotImplementedError('Analysis not implemented for {}'.format(alg['name']))

            collected_results.append({
                # Trial info
                'alg': alg['name'],
                'task': task['name'],
                'num_trajs': num_trajs,
                'run': run,
                # Expert performance
                'ex_traj_returns': ex_traj_returns,
                'ex_traj_lengths': ex_traj_lengths,
                # Learned policy performance
                'alg_traj_returns': alg_traj_returns,
                'alg_traj_lengths': alg_traj_lengths,
            })

    collected_results = pd.DataFrame(collected_results)
    with pd.HDFStore(results_full_path, 'w') as outf:
        outf['results'] = collected_results


def main():
    """ 
    This will pick one of the three phases to use, the choice of which needs to
    be supplied as a command-line argument. The phase dict maps to one of three
    *functions*, which then is used to handle the rest of its logic. I think we
    just run these in order with the same set of arguments (which basically is
    the yaml file...) except for that phase. If redoing trajectory sampling,
    delete the correct `trajs` directory. If redoing training, delete the
    correct `checkpoints` directory.

    Here's what I've run so far:

        Classic
    python scripts/im_pipeline.py pipelines/im_classic_pipeline.yaml 0_sampletrajs
    python scripts/im_pipeline.py pipelines/im_classic_pipeline.yaml 1_train
    python scripts/im_pipeline.py pipelines/im_classic_pipeline.yaml 2_eval

        Other MuJoCo
    python scripts/im_pipeline.py pipelines/im_pipeline.yaml 0_sampletrajs
    python scripts/im_pipeline.py pipelines/im_pipeline.yaml 1_train

        Reacher
    python scripts/im_pipeline.py pipelines/im_regtest_pipeline.yaml 0_sampletrajs
    python scripts/im_pipeline.py pipelines/im_regtest_pipeline.yaml 1_train

        Humanoid
    python scripts/im_pipeline.py pipelines/im_humanoid_pipeline.yaml 0_sampletrajs

    After the third script, you'll get a *single* `results.h5` file with the
    results in it.

    What's interesting is that some of the mujoco environments use a simpler
    network, just one hidden layer with 50 units and ReLUs. Why? Also, the other
    architectures here use two hidden layers with tanhs, as described in the
    paper but with 64 units instead of 100. I think I better double check what
    network options we have.

    Note: I ran into some problems with GNU screen so I had to prepend:

        screen env LD_LIBRARY_PATH=$LD_LIBRARY_PATH python scripts/im_pipeline (...)

    This will convert to screen mode, make sure it's using the correct
    environment variables, and then the python code can run after that.
    https://stackoverflow.com/questions/34779638/tensorflow-gnu-screen-for-ipython-causes-import-error

    If you want to *plot* these results, one way is to do the following:

    python scripts/showlog.py imitation_runs/classic/checkpoints/alg\=ga\,task\=cartpole\,num_trajs\=10\,run\=0.h5

    This will make several plots for that log file. However, to get a plot like
    the one in the GAIL paper, I need to make a new script.
    """
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    phases = {
        '0_sampletrajs': phase0_sampletrajs,
        '1_train': phase1_train,
        '2_eval': phase2_eval,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str)
    parser.add_argument('phase', choices=sorted(phases.keys()))
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = yaml.load(f)
    phases[args.phase](spec, args.spec)


if __name__ == '__main__':
    main()
