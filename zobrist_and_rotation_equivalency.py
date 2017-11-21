from random import randint
initial_board = (\
    ('G','G','G','G','M'),\
    ('G','G','G','G','G'),\
    ('G','G','M','G',' '),\
    ('G','G','G','G','G'),\
    ('M','G','G','G','G'))
initial_board_rot1 = (\
    ('M','G','G','G','G'),\
    ('G','G','G','G','G'),\
    ('G','G','M','G','G'),\
    ('G','G','G','G','G'),\
    ('G','G',' ','G','M'))
initial_board_rot2 = (\
    ('G','G','G','G','M'),\
    ('G','G','G','G','G'),\
    (' ','G','M','G','G'),\
    ('G','G','G','G','G'),\
    ('M','G','G','G','G'))

initial_board_rot3 = (\
    ('M','G',' ','G','G'),\
    ('G','G','G','G','G'),\
    ('G','G','M','G','G'),\
    ('G','G','G','G','G'),\
    ('G','G','G','G','M'))



board1 = (\
    (' ',' ','G',' ','M'),\
    (' ',' ',' ',' ',' '),\
    (' ',' ','M',' ',' '),\
    (' ',' ',' ',' ',' '),\
    ('M',' ',' ','G',' '))

board2 = (\
    (' ',' ',' ','G','M'),\
    (' ',' ',' ',' ',' '),\
    (' ',' ','M',' ',' '),\
    (' ',' ',' ',' ',' '),\
    ('M',' ','G',' ',' '))


board3 = (\
    ('M',' ',' ',' ',' '),\
    (' ',' ',' ',' ',' '),\
    (' ',' ','M',' ','G'),\
    (' ',' ',' ',' ',' '),\
    (' ',' ',' ',' ','M'))

board4 = (\
    ('M',' ',' ',' ',' '),\
    (' ',' ',' ',' ',' '),\
    ('G',' ','M',' ',' '),\
    (' ',' ',' ',' ',' '),\
    (' ',' ',' ',' ','M'))

board5 = (\
    ('M',' ',' ',' ',' '),\
    (' ',' ',' ',' ',' '),\
    ('G',' ','M',' ',' '),\
    (' ',' ',' ',' ',' '),\
    (' ',' ',' ',' ','M'))


ZobristTable = [\
				[[0,0],[0,0],[0,0],[0,0],[0,0]],
				[[0,0],[0,0],[0,0],[0,0],[0,0]],
				[[0,0],[0,0],[0,0],[0,0],[0,0]],
				[[0,0],[0,0],[0,0],[0,0],[0,0]],
				[[0,0],[0,0],[0,0],[0,0],[0,0]]\
				]
def init_table_zobrist():
	for i in range (0,5):
		for j in range(0,5):
			for k in range(0,2):
				#print (i)
				#print(j)
				#print(k)
				ZobristTable[i][j][k] = randint(0,2**64)
				# print (ZobristTable[i][j][k])


def compute_hash(board):
	h = 0
	for i in range (0,5):
		for j in range(0,5):
			if(board[i][j] != ' '):
				piece = 0 if board[i][j] == 'M' else 1
				h ^= ZobristTable[i][j][piece]
	return h

init_table_zobrist()
def board_piece_comparaison(p1,p2):
	piece_value = {'M': 3 , 'G' : 2, ' ': 1}
	if piece_value[p1] > piece_value[p2] : return 1
	elif piece_value[p1] < piece_value[p2] : return -1
	return 0 
def least_lexicographical_board_hash(board):
	least_lexicographical_board = board
	for rep in range(0,3):
		exitFlag = False
		board_rotation = board[::-1]
		board_rotation = tuple(zip(board_rotation[0],board_rotation[1],board_rotation[2],board_rotation[3],board_rotation[4]))
		
		for i in range(0,5):
			for j in range(0,5):
				comparaison_value = board_piece_comparaison(least_lexicographical_board[i][j], board_rotation[i][j])
				if(comparaison_value == 1):
					exitFlag = True
					break
				elif(comparaison_value == -1):
					exitFlag = True
					least_lexicographical_board = board_rotation
					break
			if(exitFlag): break
		board = board_rotation
	print ("the least value")
	for r in least_lexicographical_board:
		print(r)
	return compute_hash(least_lexicographical_board)



# for r in board5:
# 	print (r)
# print("")
# for r in board5reversed:
# 	print (r)

# print("here")
# for r in range(0,5):
# 	for c in range(0,5):
# 			print(ZobristTable[r][c])
# # print("finish here")
# # print(compute_hash(initial_board))
# print(compute_hash(board1))
# print(compute_hash(board2))
# print(compute_hash(board3))
# print(compute_hash(board4))

print(least_lexicographical_board_hash(initial_board))
print(least_lexicographical_board_hash(initial_board_rot1))
print(least_lexicographical_board_hash(initial_board_rot2))
print(least_lexicographical_board_hash(initial_board_rot3))
# for board in (initial_board,board1,board2,board3,board4):
# 	print (compute_hash(board))


