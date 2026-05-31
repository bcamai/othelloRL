import ctypes
import pyEnv as othello
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



def test_python(games_count):
    env = othello.Board(8)
    for _ in range(games_count):
        while not env.game_ended():
            moves = env.find_valid_moves(env.current_color)
            if not moves:
                env.current_color *= -1
                continue
            move = random.choice(moves)
            env.apply_move(env.current_color, move)
            env.current_color *= -1

def test_c_version(games_count):
    for _ in range(games_count):
        black_board = ctypes.c_ulonglong((1 << 28) | (1 << 35)).value
        white_board = ctypes.c_ulonglong((1 << 27) | (1 << 36)).value
        current_color = 0

        legal_moves_buf = ctypes.c_ulonglong(0)
        black_board_buf = ctypes.c_ulonglong(0)
        white_board_buf = ctypes.c_ulonglong(0)
        
        IntArray64 = ctypes.c_int * 64
        legal_moves_array_buf = IntArray64()

        skips = 0
        while skips < 2:
            c_env.FindValidMoves(black_board, white_board, 
                                 current_color, ctypes.byref(legal_moves_buf))
            legal_moves = legal_moves_buf.value
            num_valid_moves = c_env.GetScore(legal_moves)
            if num_valid_moves > 0:
                c_env.ConvertToIndexedArray(legal_moves, legal_moves_array_buf)
                valid_legal_moves = legal_moves_array_buf[:num_valid_moves]
                chosen_move = random.choice(valid_legal_moves)
                move = 1 << chosen_move
                c_env.ApplyMove(black_board, white_board,
                                current_color, move,
                                ctypes.byref(black_board_buf),
                                ctypes.byref(white_board_buf))
                black_board = black_board_buf.value
                white_board = white_board_buf.value
            else:
                skips += 1
            current_color = 1 - current_color

if __name__ == "__main__":
    GAMES = 1000

    start_python = time.time()
    test_python(GAMES)
    time_python = time.time() - start_python

    start_c = time.time()
    test_c_version(GAMES)
    time_c = time.time() - start_c

    print(f"Time (python): {time_python}, Time (C): {time_c} ")
