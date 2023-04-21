import gym
import gym_pygame
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
# env = gym.make("Pixelcopter-PLE-v0")
env = make_vec_env('AntBulletEnv-v0', n_envs=20)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# vec_env = model.get_env()
# obs = vec_env.reset()
vec_env = model.get_env()
obs = vec_env.reset()
reward_list =[]
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward)
    vec_env.render()
    reward_list.append(reward)
# torch.save(model.state_dict, "./Pixelcopter-PLE-torch")
mean = sum(reward_list) / len(reward_list)
print(mean)
model_name = "Pixelcopter-PLE-sb3_2"

model.save(model_name)
