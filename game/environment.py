import ctypes
import torch
from game.bitboard_env import Board
from game.list_env import Board as ListBoard
import random

c_env = ctypes.CDLL('./game/othello.so')
c_env.PrintBoard.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong] 
c_env.FindValidMoves.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)]
c_env.ApplyMove.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                            ctypes.c_int, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong), 
                            ctypes.POINTER(ctypes.c_ulonglong)]
c_env.ConvertToIndexedArray.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int)]
c_env.GetScore.argtypes = [ctypes.c_ulonglong] 
c_env.PlayVsRandom.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                               ctypes.c_int, ctypes.c_ulonglong,
                               ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
                               ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]
c_env.PlayTurn.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                            ctypes.c_int, ctypes.c_ulonglong,
                            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
                            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_int)]
c_env.GetWinner.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
c_env.GetWinner.restype = ctypes.c_int
device = torch.device("cpu")

MAX = 0xFFFFFFFFFFFFFFFF
WHITE = 1
BLACK = 0


def boards_to_tensor(black_board, white_board, current_color):
    my_board = white_board if current_color == WHITE else black_board
    opp_board = black_board if current_color == WHITE else white_board
    my_bits = [(my_board >> i) & 1 for i in range(64)]
    opp_bits = [(opp_board >> i) & 1 for i in range(64)]
    return torch.tensor(my_bits + opp_bits, dtype=torch.float32).to(device)

def mask_to_tensor(valid_moves_mask):
    return torch.tensor([(valid_moves_mask >> i) & 1 for i in range(64)], dtype=torch.float32).to(device)

def list_board_to_tensor(board_2d, current_color, board_size):
    # Flips board around anti-digonal so it matches boards_to_tensor
    num_nodes = board_size * board_size
    my_bits = [0.0] * num_nodes
    opp_bits = [0.0] * num_nodes
    for r in range(board_size):
        for c in range(board_size):
            val = board_2d[r][c]

            idx = (board_size - 1 - r) * board_size + (board_size - 1 - c)

            if val == current_color:
                my_bits[idx] = 1.0
            elif val == current_color * -1:
                opp_bits[idx] = 1.0
    
    return torch.tensor(my_bits + opp_bits, dtype=torch.float32).to(device)


def list_mask_to_tensor(valid_moves_list, board_size):
    num_nodes = board_size * board_size
    mask = torch.zeros(num_nodes, dtype=torch.float32).to(device)
    for r, c in valid_moves_list:
        idx = (board_size - 1 - r) * board_size + (board_size - 1 - c)
        mask[idx] = 1.0
    return mask


class C_Environment:
    def __init__(self):
        self.black_board = ctypes.c_ulonglong((1 << 28) | (1 << 35)).value
        self.white_board = ctypes.c_ulonglong((1 << 27) | (1 << 36)).value
        
        self.legal_moves_buf = ctypes.c_ulonglong(0)
        self.black_board_buf = ctypes.c_ulonglong(0)
        self.white_board_buf = ctypes.c_ulonglong(0)
        
        self.current_color = BLACK
        self.finished = False

    def reset_env(self, random_opening=False):
        self.black_board = ctypes.c_ulonglong(0x0000000810000000).value
        self.white_board = ctypes.c_ulonglong(0x0000001008000000).value
        self.current_color = BLACK
        self.finished = False

        c_env.FindValidMoves(self.black_board, self.white_board, self.current_color, ctypes.byref(self.legal_moves_buf))
        valid_moves = self.legal_moves_buf.value

        if random_opening:
            num_random_moves = random.randint(1, 5)
            for _ in range(num_random_moves):
                if valid_moves == 0:
                    break
                
                valid_indices = [i for i in range(64) if (valid_moves >> i) & 1]
                random_move = 1 << random.choice(valid_indices)
                next_color_buf = ctypes.c_int(0)
                finished_buf = ctypes.c_int(0)
                
                c_env.PlayTurn(
                    self.black_board, self.white_board, self.current_color, random_move,
                    ctypes.byref(self.black_board_buf), ctypes.byref(self.white_board_buf),
                    ctypes.byref(self.legal_moves_buf), ctypes.byref(next_color_buf), ctypes.byref(finished_buf))
                
                self.black_board = self.black_board_buf.value
                self.white_board = self.white_board_buf.value
                valid_moves = self.legal_moves_buf.value
                self.current_color = next_color_buf.value
                
                if finished_buf.value == 1:
                    self.finished = True
                    break

        return boards_to_tensor(self.black_board, self.white_board, self.current_color), mask_to_tensor(valid_moves), valid_moves, self.current_color

    def step(self, action_idx, color):
        """For Self-Play"""
        move = 1 << action_idx
        next_color_buf = ctypes.c_int(0)
        finished_buf = ctypes.c_int(0)

        c_env.PlayTurn(
            self.black_board, self.white_board, color, move,
            ctypes.byref(self.black_board_buf), ctypes.byref(self.white_board_buf),
            ctypes.byref(self.legal_moves_buf), ctypes.byref(next_color_buf), ctypes.byref(finished_buf))

        self.black_board = self.black_board_buf.value
        self.white_board = self.white_board_buf.value
        valid_moves = self.legal_moves_buf.value
        next_color = next_color_buf.value
        self.finished = (finished_buf.value == 1)

        winner = 0
        if self.finished:
            winner = c_env.GetWinner(self.black_board, self.white_board)

        return boards_to_tensor(self.black_board, self.white_board, next_color), mask_to_tensor(valid_moves), valid_moves, next_color, winner, self.finished

    def step_against(self, action_idx):
        """For playing against random bot. Agent plays as BLACK."""
        move = 1 << action_idx
        next_color_buf = ctypes.c_int(0)
        finished_buf = ctypes.c_int(0)

        c_env.PlayVsRandom(
            self.black_board, self.white_board, BLACK, move,
            ctypes.byref(self.black_board_buf), ctypes.byref(self.white_board_buf),
            ctypes.byref(self.legal_moves_buf), ctypes.byref(next_color_buf), ctypes.byref(finished_buf)
        )

        self.black_board = self.black_board_buf.value
        self.white_board = self.white_board_buf.value
        valid_moves = self.legal_moves_buf.value
        self.finished = (finished_buf.value == 1)

        reward = 0.0
        if self.finished:
            winner = c_env.GetWinner(self.black_board, self.white_board)
            if winner == -1:    
                reward = 1.0  
            elif winner == 1:   
                reward = -1.0 
            else:               
                reward = 0.0  

        return boards_to_tensor(self.black_board, self.white_board, BLACK), mask_to_tensor(valid_moves), valid_moves, reward, self.finished

class PythonBitboardEnvironment(Board):
    def __init__(self):
        super(PythonBitboardEnvironment, self).__init__()
        self.finished = False

    def reset_env(self, random_opening=False):
        self.reset()
        self.finished = False
        valid_moves = self.find_valid_move(self.board[BLACK], self.board[WHITE], BLACK)

        if random_opening:
            num_random_moves = random.randint(1, 5)
            for _ in range(num_random_moves):
                if valid_moves == 0:
                    break
                num_moves_available = self.get_score(valid_moves)
                random_move = self.get_random_move(valid_moves, num_moves_available)
                
                self.board[BLACK], self.board[WHITE], valid_moves, self.current_color, finished_flag = self.play_turn(
                    self.board[BLACK], self.board[WHITE], self.current_color, random_move)
                
                if finished_flag == 1:
                    self.finished = True
                    break

        return boards_to_tensor(self.board[BLACK], self.board[WHITE], self.current_color), mask_to_tensor(valid_moves), valid_moves, self.current_color

    def step(self, action_idx, color):
        """For selfplay"""
        move = 1 << action_idx
        
        self.board[BLACK], self.board[WHITE], valid_moves, next_turn_color, finished_flag = self.play_turn(
            self.board[BLACK], self.board[WHITE], color, move)
        
        if finished_flag == 1:
            self.finished = True
            
        winner = 0
        if self.finished:
            winner = self.get_winner(self.board[BLACK], self.board[WHITE])

        return boards_to_tensor(self.board[BLACK], self.board[WHITE], next_turn_color), mask_to_tensor(valid_moves), valid_moves, next_turn_color, winner, self.finished

    def step_against(self, action_idx):
        """For playing against random player
           Agent play as a black"""
        move = 1 << action_idx
        self.board[BLACK], self.board[WHITE] = self.apply_move(self.board[BLACK], self.board[WHITE], BLACK, move)
        
        color = WHITE
        valid_moves = self.find_valid_move(self.board[BLACK], self.board[WHITE], color)
        
        while True:
            if valid_moves == 0:
                color = BLACK
                valid_moves = self.find_valid_move(self.board[BLACK], self.board[WHITE], color)
                if valid_moves == 0:
                    self.finished = True
                    break
                else:
                    break
            else:
                num_moves = self.get_score(valid_moves)
                bot_move = self.get_random_move(valid_moves, num_moves)
                self.board[BLACK], self.board[WHITE] = self.apply_move(self.board[BLACK], self.board[WHITE], WHITE, bot_move)
                
                color = BLACK
                valid_moves = self.find_valid_move(self.board[BLACK], self.board[WHITE], color)
                if valid_moves != 0:
                    break
                
                color = WHITE
                valid_moves = self.find_valid_move(self.board[BLACK], self.board[WHITE], color)
                if valid_moves == 0:
                    self.finished = True
                    break

        reward = 0.0
        if self.finished:
            winner = self.get_winner(self.board[BLACK], self.board[WHITE])
            if winner == -1:    reward = 1.0  
            elif winner == 1:   reward = -1.0 
            else:               reward = 0.0  

        return boards_to_tensor(self.board[BLACK], self.board[WHITE], BLACK), mask_to_tensor(valid_moves), valid_moves, reward, self.finished

class PythonListEnvironment(ListBoard):
    def __init__(self, size=8):
        super(PythonListEnvironment, self).__init__(board_size=size)
        # NOTE For now agent plays only as black
        self.agent_color = -1

    def reset_env(self, random_opening=False):
        self.reset_board()
        self.current_color = self.agent_color

        valid_moves = self.find_valid_moves(self.current_color)

        if random_opening:
            num_random_moves = random.randint(1, 5)
            for _ in range(num_random_moves):
                if len(valid_moves) == 0:
                    break
                
                random_move = random.choice(valid_moves)
                self.play_turn(self.current_color, random_move)
                valid_moves = self.find_valid_moves(self.current_color)
                
                if self.finished:
                    break

        return list_board_to_tensor(self.board, self.current_color, self.board_size), list_mask_to_tensor(valid_moves, self.board_size), valid_moves, self.current_color

    def step(self, action_idx, color):
        # NOTE Changed because, list_to_board_tensor() was also changed
        # It was changed so it matches output of boards_to_tensor
        # Output of board list is flipped around anit-diagonal
        row = self.board_size - 1 - (action_idx // self.board_size)
        col = self.board_size - 1 - (action_idx % self.board_size)
        move = (row, col)

        self.play_turn(color, move)
        
        valid_moves = self.find_valid_moves(self.current_color)
        
        winner = 0
        if self.finished:
            white_pts, black_pts, _ = self.get_score()
            if black_pts > white_pts:
                winner = -1
            elif white_pts > black_pts:
                winner = 1

        return list_board_to_tensor(self.board, self.current_color, self.board_size), list_mask_to_tensor(valid_moves, self.board_size), valid_moves, self.current_color, winner, self.finished

    def step_against(self, action_idx):
        row = self.board_size - 1 - (action_idx // self.board_size)
        col = self.board_size - 1 - (action_idx % self.board_size)
        agent_move = (row, col)

        self.apply_move(self.agent_color, agent_move)
        self.current_color *= -1
        #self.play_turn(self.agent_color, agent_move)
        
        bot_color = self.agent_color * -1
        
        while not self.finished:
            bot_moves = self.find_valid_moves(bot_color)
            if len(bot_moves) > 0:
                bot_move = random.choice(bot_moves)
                self.apply_move(bot_color, bot_move)
                #self.play_turn(bot_color, bot_move)
            else:
                self.current_color = self.agent_color

            agent_moves = self.find_valid_moves(self.agent_color)
            if len(agent_moves) > 0:
                break
            else:
                self.current_color = bot_color
                if len(self.find_valid_moves(bot_color)) == 0:
                    self.finished = True
                    break

        valid_moves = self.find_valid_moves(self.agent_color)
        
        reward = 0.0
        if self.finished:
            white_pts, black_pts, _ = self.get_score()
            if black_pts > white_pts:
                reward = 1.0  
            elif white_pts > black_pts:
                reward = -1.0  
            else:
                reward = 0.0   

        return list_board_to_tensor(self.board, self.agent_color, self.board_size), list_mask_to_tensor(valid_moves, self.board_size), valid_moves, reward, self.finished
