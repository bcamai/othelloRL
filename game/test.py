import ctypes
from pythonEnvironment import Board, BLACK, WHITE
from pyEnv import Board as ListBoard
import sys

c_env = ctypes.CDLL('./othello.so') 
c_env.FindValidMoves.argtypes = [
    ctypes.c_ulonglong, ctypes.c_ulonglong,
    ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)]

c_env.ApplyMove.argtypes = [
    ctypes.c_ulonglong, ctypes.c_ulonglong,
    ctypes.c_int, ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]

MAX = 0xFFFFFFFFFFFFFFFF

def format_hex(val):
    return f"0x{val:016X}"

def bit_to_tuple(mask):
    idx = (mask.bit_length() - 1)
    row = idx % 8
    col = idx // 8
    return (row, col)

def list_moves_to_mask(moves_list):
    mask = 0
    for r, c in moves_list:
        mask |= (1 << (c * 8 + r))
    return mask

def list_board_to_masks(board_2d, size=8):
    black = 0
    white = 0
    for r in range(size):
        for c in range(size):
            if board_2d[r][c] == -1: 
                black |= (1 << (c * 8 + r))
            elif board_2d[r][c] == 1: 
                white |= (1 << (c * 8 + r))
    return black, white


def run_differential_tests(num_games):
    py_env = Board()
    list_env = ListBoard(8)
    c_moves_buf = ctypes.c_ulonglong(0)
    c_black_after_buf = ctypes.c_ulonglong(0)
    c_white_after_buf = ctypes.c_ulonglong(0)
    
    for game_idx in range(num_games):
        list_env.reset_board()
        black = 0x0000000810000000
        white = 0x0000001008000000
        color = BLACK
        passes = 0
        turn = 0
        
        while passes < 2:
            turn += 1
            py_valid_moves = py_env.find_valid_move(black, white, color)

            c_env.FindValidMoves(black, white, color, ctypes.byref(c_moves_buf))
            c_moves = c_moves_buf.value

            list_color = 1 if color == WHITE else -1

            list_moves_raw = list_env.find_valid_moves(list_color)
            list_moves_mask = list_moves_to_mask(list_moves_raw)

            #if not (py_valid_moves == c_moves == list_moves_mask):
            if (py_valid_moves & MAX) != (c_moves & MAX) or (c_moves & MAX) != (list_moves_mask & MAX):
                print(f"\n Game: {game_idx}, Turn: {turn}")
                print(f"Find valid moves error (color: {'WHITE' if color else 'BLACK'})")
                print(f"Black board: {format_hex(black)}")
                print(f"White board: {format_hex(white)}")
                print(f"Python (Bitboard) : {format_hex(py_valid_moves)}")
                print(f"Python (List)     : {format_hex(list_moves_mask)}")
                print(f"C                 : {format_hex(c_moves)}")
                sys.exit(1)
            
            if py_valid_moves == 0:
                passes += 1
                color = 1 - color
                continue
            
            passes = 0
            
            py_n_valid_moves = py_env.get_score(py_valid_moves)
            move = py_env.get_random_move(py_valid_moves, py_n_valid_moves)
            list_move_tuple = bit_to_tuple(move)

            list_env.valid_moves[list_color] = list_moves_raw
            py_black, py_white = py_env.apply_move(black, white, color, move)
            c_env.ApplyMove(black, white, color, move, 
                            ctypes.byref(c_black_after_buf), ctypes.byref(c_white_after_buf))
            c_black = c_black_after_buf.value
            c_white = c_white_after_buf.value
            
            list_env.valid_moves[list_color] = list_moves_raw
            list_env.apply_move(list_color, list_move_tuple)
            list_black, list_white = list_board_to_masks(list_env.board)
            
            #if not (py_black == c_black == list_black) or (py_white == c_white == list_white):
            if (py_black & MAX) != (c_black & MAX) or (c_black & MAX) != (list_black & MAX) or \
               (py_white & MAX) != (c_white & MAX) or (c_white & MAX) != (list_white & MAX):
                print(f"\nGame: {game_idx}, Turn: {turn}")
                print(f"Apply move error (Color: {'WHITE' if color else 'BLACK'})")
                print(f"Black board before: {format_hex(black)}")
                print(f"White board before: {format_hex(white)}")
                print(f"Applied move      : {format_hex(move)}")
                print(f"Black board after (python bitboard)   : {format_hex(py_black)}")
                print(f"White board after (python bitboard)   : {format_hex(py_white)}")
                print(f"Black board after (python list)       : {format_hex(py_black)}")
                print(f"White board after (python list)       : {format_hex(py_white)}")
                print(f"Black board after (C)                 : {format_hex(c_black)}")
                print(f"White board after (C)                 : {format_hex(c_white)}")
                sys.exit(1)
                
            black = py_black
            white = py_white
            color = 1 - color
            
        if (game_idx + 1) % 1000 == 0:
            print(f"Played {game_idx + 1}, found no error")

    print(f"\nFinished. Played {num_games} games. Found no error")

if __name__ == "__main__":
    run_differential_tests(10000)
