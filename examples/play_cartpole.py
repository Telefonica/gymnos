#
#
#   Play cartpole
#
#

from gymnos.utils import assert_dependencies

assert_dependencies([
    "gymnos[rl.policy_optimization.a2c]"
])

import gym

from gymnos.rl.policy_optimization.a2c import A2CPredictor

predictor: A2CPredictor = A2CPredictor.from_pretrained("ruben/models/a2c-cartpole")

env = gym.make("CartPole-v1")

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
