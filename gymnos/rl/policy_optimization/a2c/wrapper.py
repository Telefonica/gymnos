#
#
#   Wrapper
#
#

import gym

import torchvision.transforms.functional as F


class TorchImageWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        if len(self.observation_space.shape) == 2:
            channels = 1
            height, width = self.observation_space.shape
        else:
            height, width, channels = self.observation_space.shape

        self.observation_space = gym.spaces.Box(0, 1.0, (channels, height, width))

    def observation(self, observation):
        return F.to_tensor(observation)


def cnn_wrapper(env):
    env = TorchImageWrapper(env)
    return env
