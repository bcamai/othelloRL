class Board:
    def __init__(self, board_size):
        self.board_size = board_size
        self.finished = False
        self.current_color = -1
        self.valid_moves = {1:[], -1:[]}
        self.black_points = 0
        self.white_poits = 0
        self.board = [[0 for i in range(board_size)] for j in range(board_size)]
        self.board[board_size//2 - 1][board_size//2 - 1] = 1
        self.board[board_size//2 - 1][board_size//2] = -1
        self.board[board_size//2][board_size//2 - 1] = -1
        self.board[board_size//2][board_size//2] = 1

    def reset_board(self):
        self.finished = False
        self.current_color = -1
        self.valid_moves = {1:[], -1:[]}
        self.black_points = 0
        self.white_poits = 0
        self.board = [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        self.board[self.board_size//2 - 1][self.board_size//2 - 1] = 1
        self.board[self.board_size//2 - 1][self.board_size//2] = -1
        self.board[self.board_size//2][self.board_size//2 - 1] = -1
        self.board[self.board_size//2][self.board_size//2] = 1

    def find_valid_moves(self, current_color):
        other_color = 1
        if current_color == 1:
            other_color = -1

        valid_moves = set()

        directions = ((-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),   (1, 0), (1, 1))

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == current_color:
                    for row_dir, col_dir in directions:
                        r, c = row + row_dir, col + col_dir
                        if r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == other_color:
                            while r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == other_color:
                                r += row_dir
                                c += col_dir
                            if r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == 0:
                                valid_moves.add((r, c))
        self.valid_moves[current_color] = list(valid_moves)
        return self.valid_moves[current_color]

    def apply_move(self, current_color, position):
        if position not in self.valid_moves[current_color]:
            raise Exception("This move is invalid")

        other_color = 1
        if current_color == 1:
            other_color = -1

        directions = ((-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),   (1, 0), (1, 1))
        
        row, col = position
        self.board[row][col] = current_color

        for row_dir, col_dir in directions:
            to_swap = []
            r, c = row + row_dir, col + col_dir
            if r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == other_color:
                while r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == other_color:
                    to_swap.append((r, c))
                    r += row_dir
                    c += col_dir
                if r >= 0 and c >= 0 and r < self.board_size and c < self.board_size and self.board[r][c] == current_color:
                    for p, q in to_swap:
                        self.board[p][q] = current_color

    def get_score(self):
        white, black, empty = 0, 0, 0

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 1:
                    white += 1
                elif self.board[row][col] == -1:
                    black += 1
                else:
                    empty += 1
        return white, black, empty

    def game_ended(self):
        if len(self.find_valid_moves(self.current_color)) == 0 and len(self.find_valid_moves(-self.current_color)) == 0:
            self.finished = True
            return True
        white, black, empty = self.get_score()
        if empty == 0:
            self.finished = True
            return True
        return False

    def play_turn(self, current_color, move):
        self.apply_move(current_color, move)
        if self.game_ended():
            return

        self.current_color *= -1

        if self.current_color == current_color * -1 and len(self.valid_moves[self.current_color]) == 0:
            self.current_color *= -1

