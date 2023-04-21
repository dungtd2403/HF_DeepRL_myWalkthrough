from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import numpy as np
import math 
from collections import deque

import matplotlib.pyplot as plt
# %matplotlib inline
import stable_baselines3 
from huggingface_sb3 import package_to_hub, push_to_hub
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

# Gym
import gym
import gym_pygame

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
# import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CartPole_v1
# env_id = "CartPole-v1"
# Pixelcopter
env_id = "Pixelcopter-PLE-v0"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
# print(s_size) = 
a_size = env.action_space.n

m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.7 ]))
print(m.sample())

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        print(f'state {state}')
        # softmax return the policy probability
        probs = self.forward(state).cpu()

        m = Categorical(probs)
        print(f'm : {m}')
        
        action = m.sample()
        print('action')
        return action.item(), m.log_prob(action)



def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    best_reward = -math.inf
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset() # TODO: reset the environment
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)# TODO get the action
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action= action)# TODO: take an env step
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity 
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...
        
        # Given this formulation, the returns at each timestep t can be computed 
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order 
        # to avoid computing them multiple times)
        
        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
        
        
        ## Given the above, we calculate the returns at timestep t as: 
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed 
        ## if we were to do it from first to last.
        
        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t] ) # TODO: complete here        
       
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= best_reward:
                best_policy = policy
                best_reward = np.mean(scores_deque)
        PATH = './best_pixelcopter.pth'
    torch.save({'model_state_dict': best_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, PATH)
    torch.save(best_policy, './full_model_pixelcopter.pt')
        
        
    return scores , best_policy

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param policy: The Reinforce agent
        """
        episode_rewards = []
        for episode in range(n_eval_episodes):
            state = env.reset()
            step = 0
            done = False
            total_rewards_ep = 0
            
            for step in range(max_steps):
                action, _ = policy.act(state)
                new_state, reward, done, info = env.step(action)
                total_rewards_ep += reward
                
                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

if __name__ == '__main__':
    # cartpole_hyperparameters = {
    # "h_size": 64,
    # "n_training_episodes": 1000,
    # "n_evaluation_episodes": 10,
    # "max_t": 1000,
    # "gamma": 1.0,
    # "lr": 1e-2,
    # "env_id": env_id,
    # "state_space": s_size,
    # "action_space": a_size,
    # }
    cartpole_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
    }
    cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])
    
    
    score,cartpole_policy = reinforce(cartpole_policy,
                   cartpole_optimizer,
                   cartpole_hyperparameters["n_training_episodes"], 
                   cartpole_hyperparameters["max_t"],
                   cartpole_hyperparameters["gamma"], 
                   1000)

    mean_reward, std_reward = evaluate_agent(eval_env, 
               cartpole_hyperparameters["max_t"], 
               cartpole_hyperparameters["n_evaluation_episodes"],
               cartpole_policy)
    print(mean_reward, std_reward)
   
    