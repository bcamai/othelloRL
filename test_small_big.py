import torch
import random
import torch.nn.functional as F
from agent import ActorCriticGNN
from torch.distributions import Categorical
from game.pyEnv import Board
from game.environment import PythonListEnvironment

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def play_match(board_size, num_games, model_path="othello_selfplay_ppo_gnn.pth"):
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
    env = PythonListEnvironment(size=board_size)
    
    wins = 0
    draws = 0
    
    print(f"[{board_size}x{board_size}]  ({num_games} games) vs Random Bot")
    for game in range(num_games):
        state, mask, raw_moves, current_color = env.reset_env(random_opening=False)
        while True:
            with torch.no_grad():
                action_logits, _ = agent(state.unsqueeze(0))
                action_logits = action_logits.reshape(num_nodes)
                mask = mask.reshape(num_nodes)
                
                masked_logits = action_logits + (1.0 - mask) * -1e9
                action_idx = torch.argmax(masked_logits).item()
                #dist = Categorical(logits=masked_logits)
                #action_idx = dist.sample().item()
                
            next_state, next_mask, next_raw_moves, reward, done = env.step_against(action_idx)
            state, mask = next_state, next_mask
            
            if done:
                #white_pts, black_pts, _ = env.env.get_score()
                #print(f"Game {game+1} | Agent (Black): {black_pts} | Random Bot (White): {white_pts}")
                if reward == 1.0: wins += 1
                elif reward == 0.0: draws += 1
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
        play_match(board_size=size, num_games=games_per_size, model_path="othello_self_play_ppo_gnn.pth")
