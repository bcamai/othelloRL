#include <stdio.h>

typedef unsigned long long u64;
typedef unsigned int uint32;

#define NOT_LEFT_COL 0xFEFEFEFEFEFEFEFEllu
#define NOT_RIGHT_COL 0x7F7F7F7F7F7F7F7Fllu
#define NOT_UPPER_ROW 0x00FFFFFFFFFFFFFFllu
#define NOT_LOWER_ROW 0xFFFFFFFFFFFFFF00llu

#define WHITE 1
#define BLACK 0

// bit 0 H8 bit 63 A1
void 
PrintBoard(u64 Black, u64 White)
{
	printf("  A B C D E F G H\n");
	for(int Col = 7; Col >= 0; Col--)
	{
		printf("%d ", Col+1);
		for(int Row = 7; Row >= 0; Row--)
		{
			int Pos = Col * 8 + Row;
			u64 Mask = 1ULL << Pos;
			if(Black & Mask)
			{
				printf("b ");
			}
			else if(White & Mask)
			{
				printf("w ");
			}
			else
			{
				printf(". ");
			}
		}
		printf("\n");
	}
	printf("\n");
}

uint32 
GetScore(u64 Board)
{
	// Brian Kernighan's method of couting set bit
	uint32 score = 0;
	while(Board)
	{
		Board = Board & (Board - 1);
		score++;
	}
	return score;
}


void
ApplyMove(u64 BlackBoard, u64 WhiteBoard, 
		bool CurrentColor, u64 Move,
		u64 *BlackAfter, u64 *WhiteAfter)
{
	u64 CurrentBoard = BlackBoard;
	u64 OpponentsBoard = WhiteBoard;
	if(CurrentColor)
	{
		CurrentBoard = WhiteBoard;
		OpponentsBoard = BlackBoard;
	}

	u64 ToSwap = 0, ToSwapTemp = 0, Position;
	// Left
	Position = OpponentsBoard & (Move << 1) & NOT_LEFT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = (Position << 1) & NOT_LEFT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Right
	Position = OpponentsBoard & (Move >> 1) & NOT_RIGHT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = (Position >> 1) & NOT_RIGHT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Up
	Position = OpponentsBoard & (Move << 8) & NOT_UPPER_ROW;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = (Position << 8) & NOT_UPPER_ROW;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Down
	Position = OpponentsBoard & (Move >> 8) & NOT_LOWER_ROW;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = (Position >> 8) & NOT_LOWER_ROW;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Upper Left
	Position = OpponentsBoard & ((Move << 8) << 1) & NOT_UPPER_ROW & NOT_LEFT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = ((Position << 8) << 1) & NOT_UPPER_ROW & NOT_LEFT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Upper Right
	Position = OpponentsBoard & ((Move << 8) >> 1) & NOT_UPPER_ROW & NOT_RIGHT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = ((Position << 8) >> 1) & NOT_UPPER_ROW & NOT_RIGHT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Lower Left
	Position = OpponentsBoard & ((Move >> 8) << 1) & NOT_LOWER_ROW & NOT_LEFT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = ((Position >> 8) << 1) & NOT_LOWER_ROW & NOT_LEFT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;
	// Lower Right
	Position = OpponentsBoard & ((Move >> 8) >> 1) & NOT_LOWER_ROW & NOT_RIGHT_COL;
	while(Position & OpponentsBoard)
	{
		ToSwapTemp = ToSwapTemp | Position;
		Position = ((Position >> 8) >> 1) & NOT_LOWER_ROW & NOT_RIGHT_COL;
	}
	if(Position & CurrentBoard)
	{
		ToSwap = ToSwap | ToSwapTemp;
	}
	ToSwapTemp = 0;

	CurrentBoard = CurrentBoard | ToSwap | Move;
	OpponentsBoard = OpponentsBoard & ~ToSwap;

	if(CurrentColor)
	{
		*WhiteAfter = CurrentBoard;
		*BlackAfter = OpponentsBoard;
	}
	else
	{
		*WhiteAfter = OpponentsBoard;
		*BlackAfter = CurrentBoard;
	}
}

	void
FindValidMoves(u64 BlackBoard, u64 WhiteBoard,
							 bool CurrentColor, u64 *Moves)
{
	u64 CurrentBoard = BlackBoard;
	u64 OpponentsBoard = WhiteBoard;
	if(CurrentColor)
	{
		CurrentBoard = WhiteBoard;
		OpponentsBoard = BlackBoard;
	}

	u64 Position, Empty, ValidMoves = 0;

	Empty = ~(CurrentBoard | OpponentsBoard);

	// Left
	Position = OpponentsBoard & ((CurrentBoard << 1) & NOT_LEFT_COL);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & (Position << 1 & NOT_LEFT_COL));
		Position = OpponentsBoard & ((Position << 1) & NOT_LEFT_COL);
	}
	// Right
	Position = OpponentsBoard & ((CurrentBoard >> 1) & NOT_RIGHT_COL);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & (Position >> 1 & NOT_RIGHT_COL));
		Position = OpponentsBoard & ((Position >> 1) & NOT_RIGHT_COL);
	}
	// Up
	Position = OpponentsBoard & ((CurrentBoard << 8) & NOT_UPPER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & (Position << 8 & NOT_UPPER_ROW));
		Position = OpponentsBoard & ((Position << 8) & NOT_UPPER_ROW);
	}
	// Down
	Position = OpponentsBoard & ((CurrentBoard >> 8) & NOT_LOWER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & (Position >> 8 & NOT_LOWER_ROW));
		Position = OpponentsBoard & ((Position >> 8) & NOT_LOWER_ROW);
	}
	// Upper Left
	Position = OpponentsBoard & ((CurrentBoard << 1 << 8) & NOT_LEFT_COL & NOT_UPPER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & ((Position << 1 << 8) & NOT_UPPER_ROW));
		Position = OpponentsBoard & ((Position << 1 << 8) & NOT_LEFT_COL & NOT_UPPER_ROW);
	}
	// Upper Right
	Position = OpponentsBoard & ((CurrentBoard >> 1 << 8) & NOT_RIGHT_COL & NOT_UPPER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & ((Position >> 1 << 8) & NOT_RIGHT_COL & NOT_UPPER_ROW));
		Position = OpponentsBoard & ((Position >> 1 << 8) & NOT_RIGHT_COL & NOT_UPPER_ROW);
	}
	// Lower Left
	Position = OpponentsBoard & ((CurrentBoard << 1 >> 8) & NOT_LEFT_COL & NOT_LOWER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & ((Position << 1 >> 8) & NOT_LEFT_COL & NOT_LOWER_ROW));
		Position = OpponentsBoard & ((Position << 1 >> 8) & NOT_LEFT_COL & NOT_LOWER_ROW);
	}
	// Lower Right
	Position = OpponentsBoard & ((CurrentBoard >> 1 >> 8) & NOT_RIGHT_COL & NOT_LOWER_ROW);
	while(Position)
	{
		ValidMoves = ValidMoves | (Empty & ((Position >> 1 >> 8) & NOT_RIGHT_COL & NOT_LOWER_ROW));
		Position = OpponentsBoard & ((Position >> 1 >> 8) & NOT_RIGHT_COL & NOT_LOWER_ROW);
	}

	*Moves = ValidMoves;
}


int
main() {
	u64 BlackBoard = (1ULL << 28) | (1ULL << 35);
	u64 WhiteBoard = (1ULL << 27) | (1ULL << 36);

	int WhiteScore = GetScore(WhiteBoard);
	int BlackScore = GetScore(BlackBoard);

	printf("Init: %i %i \n", WhiteScore, BlackScore);
	u64 ValidMoves;
	FindValidMoves(BlackBoard, WhiteBoard,
								 BLACK, &ValidMoves);
	PrintBoard(BlackBoard, WhiteBoard);
	PrintBoard(ValidMoves, 0ull);
	
	ApplyMove(BlackBoard, WhiteBoard,
			0, 1ULL << 26,
			&BlackBoard, &WhiteBoard);
	WhiteScore = GetScore(WhiteBoard);
	BlackScore = GetScore(BlackBoard);	
	printf("Test move 1. %i %i \n", WhiteScore, BlackScore);
	FindValidMoves(BlackBoard, WhiteBoard,
								 WHITE, &ValidMoves);
	PrintBoard(BlackBoard, WhiteBoard);
	PrintBoard(0ull, ValidMoves);
	
	ApplyMove(BlackBoard, WhiteBoard,
			1, 1ULL << 20,
			&BlackBoard, &WhiteBoard);
	WhiteScore = GetScore(WhiteBoard);
	BlackScore = GetScore(BlackBoard);	
	printf("Test move 2. %i %i \n", WhiteScore, BlackScore);
	FindValidMoves(BlackBoard, WhiteBoard,
								 BLACK, &ValidMoves);
	PrintBoard(BlackBoard, WhiteBoard);
	PrintBoard(ValidMoves, 0ull);
	return 0;
}
