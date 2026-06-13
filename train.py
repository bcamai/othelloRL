import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from game.pythonEnvironment import Board, BLACK, WHITE

# TODO redo this file

def boards_to_tensor(black_board, white_board, current_color):
    my_board = white_board if current_color == WHITE else black_board
    opp_board = black_board if current_color == WHITE else white_board
    
    my_bits = [(my_board >> i) & 1 for i in range(64)]
    opp_bits = [(opp_board >> i) & 1 for i in range(64)]
    
    return torch.tensor(my_bits + opp_bits, dtype=torch.float32)

def mask_to_tensor(valid_moves_mask):
    return torch.tensor([(valid_moves_mask >> i) & 1 for i in range(64)], dtype=torch.float32)



class OthelloGymEnv:
    def __init__(self):
        self.env = Board()
        self.reset()

    def reset(self):
        self.black = 0x0000000810000000
        self.white = 0x0000001008000000
        self.finished = False
        
        valid_moves = self.env.find_valid_move(self.black, self.white, BLACK)
        return boards_to_tensor(self.black, self.white, BLACK), mask_to_tensor(valid_moves), valid_moves

    def step(self, action_idx):
        move = 1 << action_idx
        self.black, self.white = self.env.apply_move(self.black, self.white, BLACK, move)
        
        color = WHITE
        valid_moves = self.env.find_valid_move(self.black, self.white, color)
        
        while True:
            if valid_moves == 0:
                color = BLACK
                valid_moves = self.env.find_valid_move(self.black, self.white, color)
                if valid_moves == 0:
                    self.finished = True
                    break
                else:
                    break
            else:
                num_moves = self.env.get_score(valid_moves)
                bot_move = self.env.get_random_move(valid_moves, num_moves)
                self.black, self.white = self.env.apply_move(self.black, self.white, WHITE, bot_move)
                
                color = BLACK
                valid_moves = self.env.find_valid_move(self.black, self.white, color)
                if valid_moves != 0:
                    break
                
                color = WHITE
                valid_moves = self.env.find_valid_move(self.black, self.white, color)
                if valid_moves == 0:
                    self.finished = True
                    break

        reward = 0.0
        if self.finished:
            winner = self.env.get_winner(self.black, self.white)
            if winner == -1:    
                reward = 1.0
            elif winner == 1:   
                reward = -1.0
            else:
                reward = 0.0    

        return (boards_to_tensor(self.black, self.white, BLACK), 
                mask_to_tensor(valid_moves), 
                valid_moves, 
                reward, 
                self.finished)



class OthelloMLP(nn.Module):
    def __init__(self):
        super(OthelloMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_mask = zip(*batch)
        return (torch.stack(state), 
                torch.tensor(action, dtype=torch.long), 
                torch.tensor(reward, dtype=torch.float32), 
                torch.stack(next_state), 
                torch.tensor(done, dtype=torch.float32),
                torch.stack(next_mask))

    def __len__(self):
        return len(self.buffer)



def select_action(model, state, mask_tensor, epsilon):
    if random.random() < epsilon:
        legal_indices = torch.nonzero(mask_tensor).flatten().tolist()
        return random.choice(legal_indices)
    else:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0)).squeeze(0)
            masked_q = q_values + (1.0 - mask_tensor) * -1e9
            return torch.argmax(masked_q).item()



def train_step(model, target_model, optimizer, buffer, batch_size, gamma):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states)
        masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
        max_next_q = torch.max(masked_next_q, dim=1)[0]
        
        target_q = rewards + gamma * max_next_q * (1.0 - dones)

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



if __name__ == "__main__":
    EPISODES = 25000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 0.001
    TARGET_UPDATE = 10  
    
    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.05

    model = OthelloMLP()
    target_model = OthelloMLP()
    target_model.load_state_dict(model.state_dict()) 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = ReplayBuffer()
    env = OthelloGymEnv()

    win_history = deque(maxlen=100) 
    
    for episode in range(EPISODES):
        state, mask_tensor, raw_mask = env.reset()
        episode_loss = []
        
        while True:
            action = select_action(model, state, mask_tensor, epsilon)
            
            next_state, next_mask_tensor, next_raw_mask, reward, done = env.step(action)
            
            buffer.push(state, action, reward, next_state, float(done), next_mask_tensor)
            loss_val = train_step(model, target_model, optimizer, buffer, BATCH_SIZE, gamma=GAMMA)
            if loss_val is not None:
                episode_loss.append(loss_val)
                
            if done:
                win_history.append(1 if reward == 1.0 else 0)
                break
                
            state = next_state
            mask_tensor = next_mask_tensor

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        if (episode + 1) % 50 == 0:
            avg_win = (sum(win_history) / len(win_history)) * 100 if win_history else 0
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            print(f"Episode {episode+1}/{EPISODES} | WinRate (Last 100): {avg_win:.1f}% | Epsilon: {epsilon:.3f} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "othello_mlp.pth")
    print("\nFinished, saved as othello_mlp.pth")
