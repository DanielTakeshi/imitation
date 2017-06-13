import numpy as np
import policyopt
import gym
from gym import spaces, envs
gym.undo_logger_setup()
import logging; logging.getLogger('gym.core').addHandler(logging.NullHandler())


class RLGymSim(policyopt.Simulation):
    """
    One 'instance' of an environment, except with slightly different steps due
    to Jonathan Ho's desired action representation (for discrete action spaces).
    That might be the only difference?
    """

    def __init__(self, env_name):
        self.env = envs.make(env_name)
        self.action_space = self.env.action_space
        self.curr_obs = self.env.reset()
        self.is_done = False

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            # (JHo) We encode actions in finite spaces as an integer inside a
            # length-1 array but Gym wants the integer itself
            assert action.ndim == 1 and action.size == 1 and action.dtype in (np.int32, np.int64)
            action = action[0]
        else:
            assert action.ndim == 1 and action.dtype == np.float64

        self.curr_obs, reward, self.is_done, _ = self.env.step(action)
        return reward

    @property
    def obs(self):
        return self.curr_obs.copy()

    @property
    def done(self):
        return self.is_done

    def draw(self, track_body_name='torso'):
        self.env.render()
        if track_body_name is not None and track_body_name in self.env.model.body_names:
            self.env.viewer.cam.trackbodyid = self.env.model.body_names.index(track_body_name)

    def __del__(self):
        if self.env.viewer:
            self.env.viewer.finish()

    def reset(self):
        self.curr_obs = self.env.reset()
        self.is_done = False


def _convert_space(space):
    """ (JHo) Converts a rl-gym space to our own space representation. 
    
    Specifically, ...
    """
    if isinstance(space, spaces.Box):
        assert space.low.ndim == 1 and space.low.shape >= 1
        return policyopt.ContinuousSpace(dim=space.low.shape[0])
    elif isinstance(space, spaces.Discrete):
        return policyopt.FiniteSpace(size=space.n)
    raise NotImplementedError(space)


class RLGymMDP(policyopt.MDP):
    """
    Class representing an MDP which *uses* RLGymSim, along with the customized
    observation and action space representation. My best guess is that this is
    done as a wrapper around an environment like a normal gym `env` thingy
    (though as mentioned earlier, with the different states). The @property just
    means calling (...).obs_space or (...).action_space will return what we
    have for those attributes. I suppose the advantage is if we ever wanted to
    extend the code to add logic (error checks, etc.) but I'm confused as to why
    it's needed in this case. Note that policyopt.MDP is the super class and
    subclasses need to override such methods.
    """

    def __init__(self, env_name):
        print 'Gym version:', gym.version.VERSION
        self.env_name = env_name
        tmpsim = self.new_sim()
        self._obs_space = _convert_space(tmpsim.env.observation_space)
        self._action_space = _convert_space(tmpsim.env.action_space)
        self.env_spec = tmpsim.env.spec
        self.gym_env = tmpsim.env

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def new_sim(self, init_state=None):
        assert init_state is None
        return RLGymSim(self.env_name)
