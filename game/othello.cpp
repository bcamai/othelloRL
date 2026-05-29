#include <stdio.h>

typedef unsigned long long u64;
typedef unsigned int uint32;

#define NOT_LEFT_COL 0xFEFEFEFEFEFEFEFEllu
#define NOT_RIGHT_COL 0x7F7F7F7F7F7F7F7Fllu
#define NOT_UPPER_ROW 0x00FFFFFFFFFFFFFFllu
#define NOT_LOWER_ROW 0xFFFFFFFFFFFFFF00llu

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
ApplyMove(u64 Black, u64 White, 
		bool CurrentColor, u64 Move,
		u64* BlackAfter, u64* WhiteAfter)
{
	u64 CurrentBoard = Black;
	u64 OpponentsBoard = White;
	if(CurrentColor)
	{
		CurrentBoard = White;
		OpponentsBoard = Black;
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

int
main() {
	u64 BlackBoard = (1ULL << 28) | (1ULL << 35);
	u64 WhiteBoard = (1ULL << 27) | (1ULL << 36);

	printf("Init: \n");
	PrintBoard(BlackBoard, WhiteBoard);
	printf("Test move \n");
	ApplyMove(BlackBoard, WhiteBoard,
			0, 1ULL << 26,
			&BlackBoard, &WhiteBoard);
	PrintBoard(BlackBoard, WhiteBoard);
	printf("Test move 2 \n");
	ApplyMove(BlackBoard, WhiteBoard,
			1, 1ULL << 20,
			&BlackBoard, &WhiteBoard);
	PrintBoard(BlackBoard, WhiteBoard);

	return 0;
}
