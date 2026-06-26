import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

from game.pythonEnvironment import Board, BLACK, WHITE
from agent import ActorCriticGNN

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# NOTE Change learning for the moment self play is unstable
# Potential solution random openings or Dirichlet noise (alphazero)

def boards_to_tensor(black_board, white_board, current_color):
    my_board = white_board if current_color == WHITE else black_board
    opp_board = black_board if current_color == WHITE else white_board
    my_bits = [(my_board >> i) & 1 for i in range(64)]
    opp_bits = [(opp_board >> i) & 1 for i in range(64)]
    return torch.tensor(my_bits + opp_bits, dtype=torch.float32).to(device)

def mask_to_tensor(valid_moves_mask):
    return torch.tensor([(valid_moves_mask >> i) & 1 for i in range(64)], dtype=torch.float32).to(device)


class OthelloEnv:
    def __init__(self):
        self.env = Board()

    def reset(self):
        self.black = 0x0000000810000000
        self.white = 0x0000001008000000
        self.finished = False
        
        self.current_color = BLACK
        valid_moves = self.env.find_valid_move(self.black, self.white, self.current_color)
        return boards_to_tensor(self.black, self.white, self.current_color), mask_to_tensor(valid_moves), valid_moves, self.current_color

    def step(self, action_idx, color):
        move = 1 << action_idx
        self.black, self.white = self.env.apply_move(self.black, self.white, color, move)
        
        next_color = WHITE if color == BLACK else BLACK
        valid_moves = self.env.find_valid_move(self.black, self.white, next_color)
        
        if valid_moves == 0:
            next_color = color 
            valid_moves = self.env.find_valid_move(self.black, self.white, next_color)
            if valid_moves == 0:
                self.finished = True 
                
        winner = 0
        if self.finished:
            winner = self.env.get_winner(self.black, self.white)

        return boards_to_tensor(self.black, self.white, next_color), mask_to_tensor(valid_moves), valid_moves, next_color, winner, self.finished


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
            
        return action, action_logprob, state_value.squeeze()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_masks = torch.stack(self.buffer.masks, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach()

        advantages = rewards.detach() - old_state_values.detach()

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

    env = OthelloEnv()
    ppo_agent = PPOAgent(lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    win_history = deque(maxlen=100)
    
    for episode in range(1, max_episodes + 1):
        state, mask, raw_mask, current_color = env.reset()
        
        black_buffer = PPOBuffer()
        white_buffer = PPOBuffer()
        
        while True:
            action_tensor, logprob, val = ppo_agent.select_action(state, mask)
            action_idx = action_tensor.item()
            
            active_buffer = black_buffer if current_color == BLACK else white_buffer
            active_buffer.states.append(state)
            active_buffer.masks.append(mask)
            active_buffer.actions.append(action_tensor)
            active_buffer.logprobs.append(logprob)
            active_buffer.state_values.append(val)
            
            active_buffer.rewards.append(0.0)
            active_buffer.is_terminals.append(False)
            
            next_state, next_mask, next_raw_mask, next_color, winner, done = env.step(action_idx, current_color)

            state = next_state
            mask = next_mask
            current_color = next_color

            if done:
                
                if len(black_buffer.rewards) > 0:
                    black_buffer.is_terminals[-1] = True
                    if winner == -1:       black_buffer.rewards[-1] = 1.0   
                    elif winner == 1:    black_buffer.rewards[-1] = -1.0  
                    else:                 black_buffer.rewards[-1] = 0.0   

                if len(white_buffer.rewards) > 0:
                    white_buffer.is_terminals[-1] = True
                    if winner == 1:      white_buffer.rewards[-1] = 1.0   
                    elif winner == -1:     white_buffer.rewards[-1] = -1.0  
                    else:                 white_buffer.rewards[-1] = 0.0   

                for hist in [black_buffer, white_buffer]:
                    ppo_agent.buffer.states.extend(hist.states)
                    ppo_agent.buffer.masks.extend(hist.masks)
                    ppo_agent.buffer.actions.extend(hist.actions)
                    ppo_agent.buffer.logprobs.extend(hist.logprobs)
                    ppo_agent.buffer.state_values.extend(hist.state_values)
                    ppo_agent.buffer.rewards.extend(hist.rewards)
                    ppo_agent.buffer.is_terminals.extend(hist.is_terminals)

                win_history.append(1 if winner == -1 else 0)
                break

        if episode % update_every_n_episodes == 0:
            ppo_agent.update()

        if episode % 100 == 0:
            avg_win = (sum(win_history) / len(win_history)) * 100 if win_history else 0
            print(f"Episode {episode}/{max_episodes} | Black over white (Self-Play): {avg_win:.1f}%")

        if episode % 500 == 0:
            torch.save(ppo_agent.policy.state_dict(), "othello_self_play_ppo_gnn.pth")
