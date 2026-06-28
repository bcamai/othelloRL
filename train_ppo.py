import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
#from game.pythonEnvironment import Board, BLACK, WHITE
from game.environment import PythonBitboardEnvironment
from agent import ActorCriticGNN

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")




class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.masks = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
        self.masks.clear()


class PPOAgent:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = PPOBuffer()

        self.policy = ActorCriticGNN().to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.gcn1.parameters(), 'lr': lr_actor},
            {'params': self.policy.gcn2.parameters(), 'lr': lr_actor},
            {'params': self.policy.gcn3.parameters(), 'lr': lr_actor},
            {'params': self.policy.input_proj.parameters(), 'lr': lr_actor},
            {'params': self.policy.dummy_node_init, 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticGNN().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, mask):
        with torch.no_grad():
            action_logits, state_value = self.policy_old(state.unsqueeze(0))
            action_logits = action_logits.squeeze(0)
            
            masked_logits = action_logits + (1.0 - mask) * -1e9
            
            dist = Categorical(logits=masked_logits)
            
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        self.buffer.states.append(state)
        self.buffer.masks.append(mask)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value.squeeze())

        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # NOTE Do not normalize reward since critic output is (-1, 1)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_masks = torch.stack(self.buffer.masks, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach()

        advantages = rewards.detach() - old_state_values.detach()
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.K_epochs):
            logits, state_values = self.policy(old_states)
            
            masked_logits = logits + (1.0 - old_masks) * -1e9
            dist = Categorical(logits=masked_logits)
            
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = state_values.squeeze(-1)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()



if __name__ == "__main__":
    
    max_episodes = 25_000
    update_every_n_episodes = 20  
    K_epochs = 4                  
    eps_clip = 0.2                
    gamma = 0.99                  
    
    lr_actor = 0.0003
    lr_critic = 0.001

    env = PythonBitboardEnvironment()
    ppo_agent = PPOAgent(lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    win_history = deque(maxlen=100)

    
    for episode in range(1, max_episodes + 1):
        state, mask, raw_mask, current_color = env.reset_env(random_opening=True)

        if env.finished:
            # In case random opening resulted in finished game
            continue
        
        while True:
            action = ppo_agent.select_action(state, mask)
            
            next_state, next_mask, next_raw_mask, reward, done = env.step_against(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            state = next_state
            mask = next_mask

            if done:
                win_history.append(1 if reward == 1.0 else 0)
                break

        if episode % update_every_n_episodes == 0:
            ppo_agent.update()

        if episode % 100 == 0:
            avg_win = (sum(win_history) / len(win_history)) * 100 if win_history else 0
            print(f"Episode {episode}/{max_episodes} | WinRate (Last 100): {avg_win:.1f}%")

    torch.save(ppo_agent.policy.state_dict(), "models/othello_ppo_gnn.pth")
