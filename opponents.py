import random 
import copy

def generate_weighted_board(size):
    weights = [[1 for _ in range(size)] for _ in range(size)]

    for row in range(size):
        for col in range(size):
            from_edge_row = min(row, size - 1 - row)
            from_edge_col = min(col, size - 1 - col)

            if from_edge_row == 0 and from_edge_col == 0:
                weights[row][col] = 100

            elif (from_edge_row == 1 and from_edge_col == 0) or (from_edge_row == 0 and from_edge_col == 1):
                weights[row][col] = -50

            elif (from_edge_row == 1 and from_edge_col == 1):
                weights[row][col] = -50
                
            elif from_edge_row == 0 or from_edge_col == 0:
                weights[row][col] = 10

            elif from_edge_row == 1 or from_edge_col == 1:
                weights[row][col] = -2

            else:
                weights[row][col] = 1

    return weights

class RandomPlayer():
    def __init__(self, color):
        self.color = color
        
    def get_move(self, valid_moves, env):
        return random.choice(valid_moves)

class HeuristicPlayer():
    def __init__(self, color, board_size=8):
        self.color = color
        self.weights = generate_weighted_board(board_size)

    def get_move(self, valid_moves, env):
        best_move = None
        best_score = float('-inf')

        for move in valid_moves:
            row, col = move
            if self.weights[row][col] > best_score:
                best_score = self.weights[row][col]
                best_move = move
        
        return best_move


class Stochastic():
    def __init__(self, color, board_size = 8, randomness=0.2):
        self.color = color
        self.randomness = randomness
        self.RandomPlayer = RandomPlayer(color)
        self.HeurisiticPlayer = HeuristicPlayer(color, board_size)

    def get_move(self, valid_moves, env):
        if random.uniform(0, 1) < self.randomness:
            move = self.RandomPlayer.get_move(valid_moves, env) 
        else:
            move = self.HeurisiticPlayer.get_move(valid_moves, env)

        return move

class Greedy():
    def __init__(self, color):
        self.color = color

    def get_move(self, valid_moves, env):
        best_move = None
        max_flips = 0

        for move in valid_moves:
            sim_env = copy.deepcopy(env)

            white_points, black_points, _ = sim_env.get_score()
            points_before = white_points if self.color == 1 else black_points

            sim_env.apply_move(self.color, move)

            white_points, black_points, _ = sim_env.get_score()
            points_after = white_points if self.color == 1 else black_points

            flips = points_after - points_before

            if (flips) > max_flips:
                best_move = move
                max_flips = flips

        return best_move


class AlphaBeta():
    def __init__(self, color, board_size=8, depth=3):
        self.color = color
        self.depth = depth
        self.weights = generate_weighted_board(board_size)

    def get_move(self, valid_moves, env):
        _, best_move = self.minimax(env, self.depth, float('-inf'), float('inf'), True, self.color)
        return best_move
    
    def minimax(self, sim_env, depth, alpha, beta, max_player, current_turn_color):
        if depth == 0 or sim_env.finished:
            return self.evaluate_board(sim_env), None
        
        valid_moves =   sim_env.find_valid_moves(current_turn_color)

        if len(valid_moves) == 0:
            opp_color = -1 * current_turn_color 
            opp_moves = sim_env.find_valid_moves(opp_color)

            if len(opp_moves) == 0:
                sim_env.finished = True
                return self.evaluate_board(sim_env), None

            eval_score, _ = self.minimax(sim_env, depth-1, alpha, beta, not max_player, opp_color)
            return eval_score, None
        
        best_move = random.choice(valid_moves)

        if max_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_env = copy.deepcopy(sim_env)
                new_env.apply_move(current_turn_color, move)

                eval_score, _ = self.minimax(new_env, depth-1, alpha, beta, False, current_turn_color * -1)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_env = copy.deepcopy(sim_env)
                new_env.apply_move(current_turn_color, move)

                eval_score, _ = self.minimax(new_env, depth-1, alpha, beta, True, current_turn_color * -1)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move



    
    def evaluate_board(self, env):
        if env.finished:
            white_points, black_points, _ = env.get_score()
            if self.color == 1:
                if white_points > black_points: return 1000
                elif black_points > white_points: return -1000
                else: return 0
            else:
                if white_points < black_points: return 1000
                elif black_points < white_points: return -1000
                else: return 0

        score = 0

        for row in range(env.board_size):
            for col in range(env.board_size):
                if env.board[row][col] == self.color:
                    score += self.weights[row][col]
                elif env.board[row][col] == -1 * self.color:
                    score -= self.weights[row][col]
        
        return score


