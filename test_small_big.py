import torch
import random
import torch.nn.functional as F
from agent import ActorCriticGNN
from torch.distributions import Categorical
from game.pyEnv import Board

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def board_to_tensor(board_2d, agent_color, board_size):
    my_bits = []
    opp_bits = []
    for r in range(board_size):
        for c in range(board_size):
            val = board_2d[r][c]
            if val == agent_color:
                my_bits.append(1.0)
                opp_bits.append(0.0)
            elif val == -agent_color:
                my_bits.append(0.0)
                opp_bits.append(1.0)
            else:
                my_bits.append(0.0)
                opp_bits.append(0.0)
    
    return torch.tensor(my_bits + opp_bits, dtype=torch.float32).to(device)

def mask_to_tensor(valid_moves_list, board_size):
    num_nodes = board_size * board_size
    mask = torch.zeros(num_nodes, dtype=torch.float32).to(device)
    for r, c in valid_moves_list:
        idx = r * board_size + c
        mask[idx] = 1.0
    return mask

class DynamicOthelloEnv:
    def __init__(self, size=8):
        self.size = size
        self.env = Board(board_size=size)
        self.agent_color = -1 

    def reset(self):
        self.env.reset_board()
        self.env.current_color = self.agent_color
        
        valid_moves = self.env.find_valid_moves(self.agent_color)
        state_tensor = board_to_tensor(self.env.board, self.agent_color, self.size)
        mask_tensor = mask_to_tensor(valid_moves, self.size)
        
        return state_tensor, mask_tensor, valid_moves

    def step(self, action_idx):
        row = action_idx // self.size
        col = action_idx % self.size

        self.env.apply_move(self.agent_color, (row, col))
        self.env.current_color = self.agent_color * -1 
        
        bot_color = self.agent_color * -1
        
        while not self.env.finished:
            bot_moves = self.env.find_valid_moves(bot_color)
            
            if len(bot_moves) > 0:
                bot_move = random.choice(bot_moves)
                self.env.apply_move(bot_color, bot_move)
                self.env.current_color = self.agent_color
            else:
                self.env.current_color = self.agent_color
            
            agent_moves = self.env.find_valid_moves(self.agent_color)
            if len(agent_moves) > 0:
                break
            else:
                self.env.current_color = bot_color
                if len(self.env.find_valid_moves(bot_color)) == 0:
                    self.env.finished = True 
                    break

        _, _, empty = self.env.get_score()
        if empty == 0:
            self.env.finished = True

        reward = 0.0
        if self.env.finished:
            white_pts, black_pts, _ = self.env.get_score()
            if black_pts > white_pts:
                reward = 1.0  
            elif black_pts < white_pts:
                reward = 0.0  
            else:
                reward = 0.5  

        next_valid_moves = self.env.find_valid_moves(self.agent_color)
        state_tensor = board_to_tensor(self.env.board, self.agent_color, self.size)
        mask_tensor = mask_to_tensor(next_valid_moves, self.size)

        return state_tensor, mask_tensor, next_valid_moves, reward, self.env.finished


def play_match(board_size, num_games, model_path="othello_ppo_gnn.pth"):
    agent = ActorCriticGNN(board_size=board_size).to(device)
    num_nodes = board_size * board_size
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        keys_to_remove = [k for k in state_dict.keys() if 'adj_matrix' in k]
        for k in keys_to_remove:
            del state_dict[k]
            
        agent.load_state_dict(state_dict, strict=False)
        print(f"[{board_size}x{board_size}] Loaded model.")
    except Exception as e:
        print(f"[{board_size}x{board_size}] Error '{model_path}'.")

    agent.eval() 
    env = DynamicOthelloEnv(size=board_size)
    
    wins = 0
    draws = 0
    
    print(f"[{board_size}x{board_size}]  ({num_games} games) vs Random Bot")
    for game in range(num_games):
        state, mask, raw_moves = env.reset()
        
        while True:
            with torch.no_grad():
                action_logits, _ = agent(state.unsqueeze(0))
                action_logits = action_logits.reshape(num_nodes)
                mask = mask.reshape(num_nodes)
                
                masked_logits = action_logits + (1.0 - mask) * -1e9
                #action_idx = torch.argmax(masked_logits).item()
                dist = Categorical(logits=masked_logits)
                action_idx = dist.sample().item()
                
            next_state, next_mask, next_raw_moves, reward, done = env.step(action_idx)
            state, mask = next_state, next_mask
            
            if done:
                #white_pts, black_pts, _ = env.env.get_score()
                #print(f"Game {game+1} | Agent (Black): {black_pts} | Random Bot (White): {white_pts}")
                
                if reward == 1.0: wins += 1
                elif reward == 0.5: draws += 1
                break
                
    winrate = (wins / num_games) * 100
    print("")
    print(f" Board: {board_size}x{board_size}:")
    print(f" Win: {wins} | Draw: {draws} | Loss: {num_games - wins - draws}")
    print(f" WinRate: {winrate:.1f}%")
    print("")


if __name__ == "__main__":
    games_per_size = 100
    
    for size in [8, 10, 12, 16, 24, 32]:
        play_match(board_size=size, num_games=games_per_size, model_path="othello_ppo_gnn.pth")
