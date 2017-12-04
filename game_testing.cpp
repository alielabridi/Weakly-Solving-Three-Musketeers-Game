#include <bits/stdc++.h>

using namespace std;

char initial_board[][] =    {{'G','G','G','G','M'},
						    {'G','G','G','G','G'},
						    {'G','G','M','G','G'},
						    {'M','G','G','G','G'},
						    {'G',' ','G','G','G'}};
long long ZobristTable[][][] = {
				                {{0,0},{0,0},{0,0},{0,0},{0,0}},
				                {{0,0},{0,0},{0,0},{0,0},{0,0}},
				                {{0,0},{0,0},{0,0},{0,0},{0,0}},
				                {{0,0},{0,0},{0,0},{0,0},{0,0}},
				                {{0,0},{0,0},{0,0},{0,0},{0,0}}
				               }
int orthogonal_moves[][] = {{1,0},{-1,0},{0,1},{0,-1}};

void init_table_zobrist(){
	for(int i = 0 ; i < 5;  i++)
		for(int j  = 0 ; j < 5 ; j++)
			for(int k = 0 ; k < 2; k++)
				/*TODO: should be a random number*/
				ZobristTable[i][j][k] = 100000+i+k+j;
}

void compute_hash(char board[5][5]){
	long long h = 0;
	for(int i = 0 ; i < 5 ; i++)
		for(int j = 0 ; j < 5 ; j++)
			if(board[i][j] != ' '){
				int piece = board[i][j] == 'M'? 0 : 1;
				h ^= ZobristTable[i][j][piece];
			}
	return h;
}

int board_piece_comparaison(char p1, char p2){
	if(p1 == p2) return 0;
	if(p1 == 'M' && (p2 == 'G' || p2 == ' ')) return 1;
	if(p1 == 'G' && p2 == ' ') return 1;
	return -1;
}

long long least_lexicographical_board_hash(char board[5][5]){
	// TODO: implement it
}

bool within_board_range(int x, int y){
	return (y < 5 && y >= 0 && x < 5 && x >= 0);
} 

class ThreeMusketeers{

	bool terminal_test(Gamestate s){
		return (s.utility == 0 ? 0 : 1);
	}

	char to_move(Gamestate s){
		return s.to_move;
	}

	/*TODO: result, action methods*/

	
	int compute_utility(char board[5][5], char to_move, int muskteers_pos[3][2]){
		/*same row: guardmen winning*/
		if(muskteers_pos[0][0] == muskteers_pos[1][0] && muskteers_pos[1][0] == muskteers_pos[2][0])
			return -1;
		/*same column: guardmen winning*/
		if(muskteers_pos[0][1] == muskteers_pos[1][1] && muskteers_pos[1][1] == muskteers_pos[2][1])
			return -1;
		/*still possible moves (nobody wins yet)*/
		for(int i = 0 ; i < 3 ; i ++){
			for(int j = 0 ; j < 4 ; j++){
				if(within_board_range(muskteers_pos[i][0]+orthogonal_moves[j][0],
									muskteers_pos[i][1]+orthogonal_moves[j][1])
									&& board[muskteers_pos[i][0]+orthogonal_moves[j][0]][muskteers_pos[i][1]+orthogonal_moves[j][1]] == 'G')
				return 0;
			}
		}
		return 1;

	}
	int utility(Gamestate s, char player){
		return player == 'M' ? s.utility : -s.utility;
	}


}




int main(){



	return 0;
}