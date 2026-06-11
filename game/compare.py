import ctypes
import pyEnv as othello
import pythonEnvironment as py_env
import random
import time

c_env = ctypes.CDLL("./othello.so")
c_env.PrintBoard.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong] 
c_env.FindValidMoves.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                 ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)] 
c_env.ApplyMove.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                            ctypes.c_int, ctypes.c_ulonglong,
                            ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)] 
c_env.ConvertToIndexedArray.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int)]
c_env.GetScore.argtypes = [ctypes.c_ulonglong] 
c_env.PlayVsRandom.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                               ctypes.c_int, ctypes.c_ulonglong,
                               ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
                               ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]



def test_python(games_count):
    env = othello.Board(8)
    for _ in range(games_count):
        env.reset_board()
        while not env.game_ended():
            moves = env.find_valid_moves(env.current_color)
            if not moves:
                env.current_color *= -1
                continue
            move = random.choice(moves)
            env.apply_move(env.current_color, move)
            env.current_color *= -1

def test_python_bitboard(games_count):
    env = py_env.Board()
    for _ in range(games_count):
        env.reset()
        black_board = env.board[py_env.BLACK]
        white_board = env.board[py_env.WHITE]
        current_color = py_env.BLACK
        finished = 0

        while not finished:
            valid_moves = env.find_valid_move(black_board, white_board, current_color)
            n_valid_moves = env.get_score(valid_moves)
            move = env.get_random_move(valid_moves, n_valid_moves)
            black_board, white_board, valid_moves, current_color, finished = \
                    env.play_vs_random(black_board, white_board, current_color, move)
    


def test_c_version(games_count):
    for _ in range(games_count):
        black_board = ctypes.c_ulonglong((1 << 28) | (1 << 35)).value
        white_board = ctypes.c_ulonglong((1 << 27) | (1 << 36)).value
        current_color = ctypes.c_int(0)
        finished = ctypes.c_int(0)

        legal_moves_buf = ctypes.c_ulonglong(0)
        black_board_buf = ctypes.c_ulonglong(0)
        white_board_buf = ctypes.c_ulonglong(0)
        
        IntArray64 = ctypes.c_int * 64
        legal_moves_array_buf = IntArray64()

        c_env.FindValidMoves(black_board, white_board, 
                             current_color, ctypes.byref(legal_moves_buf))
        legal_moves = legal_moves_buf.value
        num_valid_moves = c_env.GetScore(legal_moves)
        c_env.ConvertToIndexedArray(legal_moves, legal_moves_array_buf)
        valid_legal_moves = legal_moves_array_buf[:num_valid_moves]
        chosen_move = random.choice(valid_legal_moves)
        move = 1 << chosen_move
        c_env.PlayVsRandom(black_board, white_board,
                        current_color.value, move,
                        ctypes.byref(black_board_buf), ctypes.byref(white_board_buf),
                        ctypes.byref(legal_moves_buf), ctypes.byref(current_color),
                        ctypes.byref(finished))
        black_board = black_board_buf.value
        white_board = white_board_buf.value
        
        while(not finished.value):
            legal_moves = legal_moves_buf.value
            c_env.ConvertToIndexedArray(legal_moves, legal_moves_array_buf)
            num_valid_moves = c_env.GetScore(legal_moves)
            valid_legal_moves = legal_moves_array_buf[:num_valid_moves]
            chosen_move = random.choice(valid_legal_moves)
            move = 1 << chosen_move
            c_env.PlayVsRandom(black_board, white_board,
                            current_color.value, move,
                            ctypes.byref(black_board_buf), ctypes.byref(white_board_buf),
                            ctypes.byref(legal_moves_buf), ctypes.byref(current_color),
                            ctypes.byref(finished))
            black_board = black_board_buf.value
            white_board = white_board_buf.value

if __name__ == "__main__":
    GAMES = 1000

    start_python = time.time()
    test_python(GAMES)
    time_python = time.time() - start_python

    start_python_bitboard = time.time()
    test_python_bitboard(GAMES)
    time_python_bitboard = time.time() - start_python_bitboard

    start_c = time.time()
    test_c_version(GAMES)
    time_c = time.time() - start_c

    print(f"Time (python): {time_python},Time (python bitboard): {time_python_bitboard}, Time (C): {time_c} ")
