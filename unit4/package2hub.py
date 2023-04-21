import gym
import gym_pygame
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import pybullet_envs
import panda_gym
from huggingface_sb3 import package_to_hub

# PLACE the variables you've just defined two cells above
# Define the name of the environment


# TODO: Define the model architecture we used
model_architecture = "A2C"

## Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
## CHANGE WITH YOUR REPO ID
repo_id = "dungtd2403/Pixelcopter-PLE-V0-SB"  # Change with your repo id, you can't push with mine 😄

## Define the commit message
commit_message = "Upload A2C Pixelcopter-PLE-sb3 trained agent"

# Create the evaluation env
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])

# PLACE the package_to_hub function you've just filled here
# model_path = "./Pixelcopter-PLE-sb3/policy.pth"
# package_to_hub(
#     model=model,  # Our trained model
#     model_name=model_name,  # The name of our trained model
#     model_architecture=model_architecture,  # The model architecture we used: in our case PPO
#     env_id=env_id,  # Name of the environment
#     eval_env=eval_env,  # Evaluation Environment
#     repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
#     commit_message=commit_message,
# )
# env = make_vec_env('Pixelcopter-PLE-v0', n_envs=50)

# model = A2C("MlpPolicy", env, verbose=1)
env_id = "AntBulletEnv-v0"
eval_env = make_vec_env(env_id, n_envs=1)
model = A2C.load('AntBulletEnv-v0')


# package_to_hub(
#     repo_id=repo_id,
#     filename="./Pixelcopter-PLE-sb3/policy.pth",
#     commit_message=commit_message,
# )
package_to_hub(model=model,
             model_name="AntBulletEnv-v0",
             model_architecture="A2C",
             env_id=env_id,
             eval_env=eval_env,
             repo_id="dungtd2403/AntBulletEnv-v0",
             commit_message="Test commit")