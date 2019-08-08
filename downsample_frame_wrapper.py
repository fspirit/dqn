import gym


class DownsampleFrameWrapper(gym.Wrapper):

    def __init__(self, env, downsample_frame_fn):
        gym.Wrapper.__init__(self, env)
        self.downsample_frame_fn = downsample_frame_fn

    def reset(self):
        ob = self.env.reset()
        return self.downsample_frame_fn(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return self.downsample_frame_fn(ob), reward, done, info
