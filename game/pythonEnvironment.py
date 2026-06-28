import random

NOT_LEFT_COL = 0xFEFEFEFEFEFEFEFE
NOT_RIGHT_COL = 0x7F7F7F7F7F7F7F7F
NOT_LOWER_ROW = 0x00FFFFFFFFFFFFFF
NOT_UPPER_ROW = 0xFFFFFFFFFFFFFF00
MAX = 0xFFFFFFFFFFFFFFFF

WHITE = 1
BLACK = 0

class Board:
    def __init__(self):
        self.board = [0, 0]
        self.board[BLACK] = 0x0000000810000000
        self.board[WHITE] = 0x0000001008000000
        # White 1, Black 0
        self.current_color = 0

    def reset(self):
        self.board[BLACK] = 0x0000000810000000
        self.board[WHITE] = 0x0000001008000000
        self.current_color = 0

    def print_board(self, board):
        print("  A B C D E F G H")
        position = 1
        for col in range(8):
            print(f"{col + 1} ", end="")
            for row in range(8):
                if(position & board):
                    print("b ", end="")
                elif(position & board):
                    print("w ", end="")
                else:
                    print(". ", end="")
                position = position << 1
            print("")
        print("")

    def get_score(self, board):
        score = 0;
        while(board):
            board = board & (board - 1)
            score += 1
        return score

    def get_winner(self, black_board, white_board):
        white_score = self.get_score(white_board)
        black_score = self.get_score(black_board)
        if(white_score > black_score):
            return 1
        elif(black_score > white_score):
            return -1
        else:
            return 0

    def convert_to_array(self, board, current_color, processed_board):
        idx = 0
        color = -1
        if current_color:
            color = 1
        while(board):
            if(board & 1):
                processed_board[idx] = 1 * color
            board = board >> 1
            idx += 1

    def convert_to_indexed_array(self, board):
        board_idx = 0
        idx = 0
        processed_board = []
        while(board):
            if (board & 1):
                processed_board.append(board_idx)
                idx += 1
            board = board >> 1
            board_idx += 1

    def apply_move(self, black_board, white_board,
                   current_color, move):
        current_board = black_board
        opponent_board = white_board
        if(current_color):
            current_board = white_board
            opponent_board = black_board
        to_swap = 0
        # Left
        to_swap_temp = 0
        position = opponent_board & (move << 1) & NOT_LEFT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position << 1) & NOT_LEFT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Right
        to_swap_temp = 0
        position = opponent_board & (move >> 1) & NOT_RIGHT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position >> 1) & NOT_RIGHT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Up
        to_swap_temp = 0
        position = opponent_board & (move << 8) & NOT_UPPER_ROW
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position << 8) & NOT_UPPER_ROW
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Lower
        to_swap_temp = 0
        position = opponent_board & (move >> 8) & NOT_LOWER_ROW
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position >> 8) & NOT_LOWER_ROW
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Upper left
        to_swap_temp = 0
        position = opponent_board & (move << 9) & NOT_UPPER_ROW & NOT_LEFT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position << 9) & NOT_UPPER_ROW & NOT_LEFT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Upper right
        to_swap_temp = 0
        position = opponent_board & (move << 7) & NOT_UPPER_ROW & NOT_RIGHT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position << 7) & NOT_UPPER_ROW & NOT_RIGHT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Lower left
        to_swap_temp = 0
        position = opponent_board & (move >> 7) & NOT_LOWER_ROW & NOT_LEFT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position >> 7) & NOT_LOWER_ROW & NOT_LEFT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp
        # Lower right
        to_swap_temp = 0
        position = opponent_board & (move >> 9) & NOT_LOWER_ROW & NOT_RIGHT_COL
        while(position & opponent_board):
            to_swap_temp = to_swap_temp | position
            position = (position >> 9) & NOT_LOWER_ROW & NOT_RIGHT_COL
        if(position & current_board):
            to_swap = to_swap | to_swap_temp

        current_board = current_board | to_swap | move
        opponent_board = opponent_board & (~to_swap & MAX)

        current_board &= MAX
        opponent_board &= MAX

        # return black_board, white_board
        if(current_color == 1):
            return opponent_board, current_board
        else:
            return current_board, opponent_board

    def find_valid_move(self, black_board,
                        white_board, current_color):
            current_board = white_board if current_color else black_board
            opponent_board = black_board if current_color else white_board

            valid_moves = 0
            empty = (~(current_board | opponent_board)) & MAX
            # Left
            position = opponent_board & ((current_board << 1) & NOT_LEFT_COL)
            while position:
                valid_moves |= (empty & ((position << 1) & NOT_LEFT_COL))
                position = opponent_board & ((position << 1) & NOT_LEFT_COL)
            # Right
            position = opponent_board & ((current_board >> 1) & NOT_RIGHT_COL)
            while position:
                valid_moves |= (empty & ((position >> 1) & NOT_RIGHT_COL))
                position = opponent_board & ((position >> 1) & NOT_RIGHT_COL)
            # Up
            position = opponent_board & ((current_board << 8) & NOT_UPPER_ROW)
            while position:
                valid_moves |= (empty & ((position << 8) & NOT_UPPER_ROW))
                position = opponent_board & ((position << 8) & NOT_UPPER_ROW)
            # Down
            position = opponent_board & ((current_board >> 8) & NOT_LOWER_ROW)
            while position:
                valid_moves |= (empty & ((position >> 8) & NOT_LOWER_ROW))
                position = opponent_board & ((position >> 8) & NOT_LOWER_ROW)
            # Upper Left
            position = opponent_board & ((current_board << 9) & NOT_LEFT_COL & NOT_UPPER_ROW)
            while position:
                valid_moves |= (empty & ((position << 9) & NOT_LEFT_COL & NOT_UPPER_ROW))
                position = opponent_board & ((position << 9) & NOT_LEFT_COL & NOT_UPPER_ROW)
            # Upper Right
            position = opponent_board & ((current_board << 7) & NOT_RIGHT_COL & NOT_UPPER_ROW)
            while position:
                valid_moves |= (empty & ((position << 7) & NOT_RIGHT_COL & NOT_UPPER_ROW))
                position = opponent_board & ((position << 7) & NOT_RIGHT_COL & NOT_UPPER_ROW)
            # Lower Left
            position = opponent_board & ((current_board >> 7) & NOT_LEFT_COL & NOT_LOWER_ROW)
            while position:
                valid_moves |= (empty & ((position >> 7) & NOT_LEFT_COL & NOT_LOWER_ROW))
                position = opponent_board & ((position >> 7) & NOT_LEFT_COL & NOT_LOWER_ROW)
            # Lower Right
            position = opponent_board & ((current_board >> 9) & NOT_RIGHT_COL & NOT_LOWER_ROW)
            while position:
                valid_moves |= (empty & ((position >> 9) & NOT_RIGHT_COL & NOT_LOWER_ROW))
                position = opponent_board & ((position >> 9) & NOT_RIGHT_COL & NOT_LOWER_ROW)

            return valid_moves

    def play_turn(self, black_board, white_board,
                  current_color, move):
            finished = 0
            next_turn_color = current_color
            
            black_after, white_after = self.apply_move(black_board, white_board, current_color, move)
            current_color = 1 - current_color
            
            valid_moves = self.find_valid_move(black_after, white_after, current_color)
            if not valid_moves:
                current_color = 1 - current_color
                valid_moves = self.find_valid_move(black_after, white_after, current_color)
                if not valid_moves:
                    finished = 1
                else:
                    next_turn_color = current_color
            else:
                next_turn_color = current_color
                
            return black_after, white_after, valid_moves, next_turn_color, finished

    def get_random_move(self, valid_moves, num_moves):
            random_index = random.randint(0, num_moves - 1)
            chosen_move = 0

            for _ in range(random_index + 1):
                chosen_move = valid_moves & (-valid_moves & MAX)
                valid_moves &= (valid_moves - 1) & MAX
                
            return chosen_move

    def play_vs_random(self, black_board, white_board, 
                       current_color, move):
            finished = 0
            next_turn_color = current_color

            black_after, white_after = self.apply_move(black_board, white_board, current_color, move)
            current_color = 1 - current_color
            
            valid_moves = self.find_valid_move(black_after, white_after, current_color)

            if not valid_moves:
                current_color = 1 - current_color
                valid_moves = self.find_valid_move(black_after, white_after, current_color)
                if not valid_moves:
                    finished = 1
                else:
                    next_turn_color = current_color
                    return black_after, white_after, valid_moves, next_turn_color, finished
            else:
                num_valid_moves = self.get_score(valid_moves)
                chosen_move = self.get_random_move(valid_moves, num_valid_moves)
                
                black_after, white_after = self.apply_move(black_after, white_after, current_color, chosen_move)
                current_color = 1 - current_color
                
                valid_moves = self.find_valid_move(black_after, white_after, current_color)
                
                while not valid_moves:
                    current_color = 1 - current_color
                    valid_moves = self.find_valid_move(black_after, white_after, current_color)
                    if not valid_moves:
                        finished = 1
                        break
                    else:
                        num_valid_moves = self.get_score(valid_moves)
                        chosen_move = self.get_random_move(valid_moves, num_valid_moves)
                        black_after, white_after = self.apply_move(black_after, white_after, current_color, chosen_move)
                        current_color = 1 - current_color
                        valid_moves = self.find_valid_move(black_after, white_after, current_color)
                        
                next_turn_color = current_color

            return black_after, white_after, valid_moves, next_turn_color, finished
