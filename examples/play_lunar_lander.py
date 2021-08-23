#
#
#   Play cartpole
#
#

from gymnos.utils import assert_dependencies

assert_dependencies([
    "gymnos[rl.value_optimization.dqn]"
])

import gym

from gymnos.rl.value_optimization.dqn import DQNPredictor

predictor = DQNPredictor.from_pretrained("ruben/models/dqn-lunar-lander")

env = gym.make("LunarLander-v2")

done = False
obs = env.reset()

episode_steps = 0
episode_reward = 0.0

while not done:
    env.render()

    action, _ = predictor.predict(obs)
    obs, reward, done, info = env.step(action)

    episode_steps += 1
    episode_reward += reward

print(f"Episode reward: {episode_reward}")
print(f"Episode steps: {episode_steps}")
