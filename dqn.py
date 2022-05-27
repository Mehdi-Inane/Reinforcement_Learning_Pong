import random
import numpy
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Store Transitions
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


steps_done = 0


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.steps_done = 0
        self.conv1  =  nn.Conv2d( 4 , 32 , kernel_size = 8 , stride = 4 , padding = 0 )
        self.conv2  =  nn.Conv2d( 32 , 64 , kernel_size = 4 , stride = 2 , padding = 0 )
        self.conv3  =  nn.Conv2d ( 64 , 64 , kernel_size = 3 , stride = 1 , padding = 0 )
        self.fc1  =  nn.Linear( 3136 , 512 )
        self.fc2  =  nn.Linear( 512 , self.n_actions )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1,3136)
        x = self.relu(self.fc1(x))
        x = self.fc2(x) 
        x=self.flatten(x)
        return x.view(2,-1)

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        global steps_done
        action = []
        if exploit:
            with torch.no_grad():
                for state in observation:
                    action.append(self.forward(state).max(0)[1].item())
                return torch.tensor(action)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / 200)
        steps_done += 1
        for state in observation:
            if random.random() <= eps_threshold:
                action_index = random.randrange(0, self.n_actions)
                action.append(action_index)
            else:
                action.append(self.forward(state).max(0)[1].item())
        return torch.tensor(action)


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!

    transitions = memory.sample(dqn.batch_size)
    Transition = namedtuple('Transition',
                            ('obs', 'action', 'next_obs', 'reward'))
    batch = Transition(*transitions)
    non_final_mask = torch.tensor(tuple(map(lambda s: not isinstance(s, numpy.ndarray),
                                            batch.next_obs)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_obs if not isinstance(s, numpy.ndarray)])
    state_batch = torch.cat(batch.obs)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn(state_batch).gather(1, action_batch.view(1,dqn.batch_size))

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

    next_state_values = torch.zeros(dqn.batch_size, device=device)
    #print(target_dqn(non_final_next_states).detach().max(0)[0].shape)
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).detach().max(0)[0]
    # Compute the expected Q values
    q_value_targets = (next_state_values * dqn.gamma) + reward_batch
    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss.item()
