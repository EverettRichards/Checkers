import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
from IPython.display import clear_output

# TODO: Implement AlphaZero

cmap = ListedColormap(['firebrick', 'lemonchiffon', 'darkseagreen', 'midnightblue', 'goldenrod'])
bounds = [-4, -2, 0, 2, 4, 6]
norm = BoundaryNorm(bounds, cmap.N)

class GameState:
    # n : side length of the (Square) checker board. Must be even. 8 by default.
    # r : number of rows to populate with checkers on each side. 3 by default.
    def __init__(self,n=8,r=3):
        n = n + n % 2 # Enforce even n
        self.n,self.r = n,r

        self.board = np.zeros((n,n),dtype=np.int8)
        for i in range(r):
            for j in range(int(n/2)):
                self.board[i][j*2 + i%2] = 1
                self.board[n-1-i][j*2 + i%2 - 1] = -1

        self.num_red = int(n*r/2)
        self.num_blue = int(n*r/2)
        plt.ion()
        plt.show(block=False)
    
    def __str__(self):
        return str(self.board)
    
    def plot(self,x=8,pad=1,crown_size=2,wait_time_ms=1000,last_move=None):
        # x: square size
        # p: padding size
        plt.cla()
        n = self.n
        modified_board = np.zeros((n*x,n*x),dtype=np.int8)
        # Grey/white checkerboard pattern
        for i in range(n):
            for j in range(n):
                modified_board[i*x:i*x+x, j*x:j*x+x] = -((i+j)%2-0.5)*2
        # Golden border for last move
        if last_move:
            i,j = last_move[0],last_move[1]
            modified_board[i*x:i*x+x, j*x:j*x+x] = 5
        # Red/black pieces
        for i in range(n):
            for j in range(n):
                current_piece, is_king = self.getPiece(i,j)
                if current_piece: # not 0 --> True?
                    modified_board[i*x+pad:i*x+x-pad, j*x+pad:j*x+x-pad] = current_piece * 3
                    if is_king:
                        cpad = (x - crown_size) // 2
                        modified_board[i*x+cpad:i*x+x-cpad, j*x+cpad:j*x+x-cpad] = 5
        plt.imshow(modified_board,cmap=cmap, norm=norm, origin="upper")
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        plt.title(f"{self.n}x{self.n} Checkerboard, r={self.r}. {self.howManyLeft(-1)} red / {self.howManyLeft(1)} blue remain.")
        plt.draw()
        plt.pause(wait_time_ms/1000)

    def getPlayer(self,p):
        # ... because kings work differently
        if p>0:
            return 1
        elif p<0:
            return -1
        else:
            return 0
    
    def howManyLeft(self,p):
        # p: player (+1 = blue, -1 = red)
        if p > 0:
            return self.num_blue
        else:
            return self.num_red
        
    def decrementPlayer(self,p):
        #p: player (+1 = blue, -1 = red)
        winner = 0
        if p > 0:
            self.num_blue -= 1
        else:
            self.num_red -= 1

        if self.num_blue == 0 or self.num_red == 0:
            winner = -self.getPlayer(p)
        
        return winner
    
    def getPiece(self,i,j):
        if i < 0 or j < 0:
            return -10,None
        elif i >= self.n or j >= self.n:
            return -11,None
        else:
            #getPlayer normalization is justified because this function is only used to see if a space is taken, and by whom
            return self.getPlayer(self.board[i][j]), abs(self.board[i][j])==2 # is it a king?

    def getPieceMoveOptions(self,i,j):
        p,is_king = self.getPiece(i,j)
        directions = [-1,1] if is_king else [1]

        output = []
        can_jump = False # If you *can* jump, you *must* jump

        for LR in [-1,1]: # You can go left OR right
            for DIR in directions: # Kings can go forward OR backward
                i1,i2,j1,j2 = i+p*DIR,i+2*p*DIR,j+LR,j+2*LR
                
                diag_piece,_ = self.getPiece(i1,j1)
                if diag_piece < -1: continue # Out of bounds

                if diag_piece == -p:
                    # enemy! query further
                    next_diag_piece,_ = self.getPiece(i2,j2)
                    if next_diag_piece < -1: continue # Out of bounds
                    if next_diag_piece == 0:
                        # can successfully jump! do something...
                        if not can_jump:
                            can_jump = True
                            output = []
                        output.append([(i,j),(i2,j2),-p])
                    else:
                        # jump is blocked.
                        pass
                elif diag_piece == 0 and not can_jump:
                    # i've got a blank space, baby
                    output.append([(i,j),(i1,j1),0])
        return output,can_jump

    def getLegalMoves(self,p,double_jump_piece=(-1,-1)):
        # p: player (+1 = blue, -1 = red)
        num_queried = 0
        n = self.n
        board = self.board
        legal_moves = [] # elements look like [start (i,j), end (i,j), move type, end(i,j), move type, ...] (0 = move, 1 = capture a blue, -1 = capture a red)
        any_can_jump = False

        double_jump_mode = double_jump_piece[0] != -1

        for i in range(n):
            if double_jump_mode and i != double_jump_piece[0]:
                continue
            for j in range(n):
                if double_jump_mode and j != double_jump_piece[1]:
                    continue
                current_piece, is_king = self.getPiece(i,j)
                if current_piece == p:
                    num_queried += 1
                    move_options,piece_can_jump = self.getPieceMoveOptions(i,j)
                    if len(move_options) > 0:
                        if piece_can_jump and not any_can_jump:
                            any_can_jump = True
                            legal_moves = move_options.copy()
                        elif any_can_jump and not piece_can_jump:
                            pass # Only non-capture moves available for the piece, but some captures can be done elsewhere
                        else:
                            legal_moves.extend(move_options) # Concatenate the new move options
                    if num_queried >= self.howManyLeft(p):
                        break
            if num_queried >= self.howManyLeft(p):
                    break
        
        return legal_moves,any_can_jump
        
    def performMove(self,move):
        # assume that the move is well-defined and valid as a precondition
        i0,j0,i1,j1 = move[0][0],move[0][1],move[1][0],move[1][1]
        p,is_king = self.getPiece(i0,j0)
        action = move[2]
        self.board[i1][j1] = self.board[i0][j0] # advance the piece
        self.board[i0][j0] = 0 # remove the shadow of the moved piece
        if action != 0:
            i2,j2 = (i0+i1)//2,(j0+j1)//2
            self.board[i2][j2] = 0 # remove the captured piece
            self.decrementPlayer(action)
        if not is_king and ((p==1 and i1==self.n-1) or (p==-1 and i1==0)):
            self.board[i1][j1] *= 2

    def playRandomGame(self,max_turns=100,wait_time_ms=1000,debug=False):
        p = 1
        turn = 1
        winner = 0
        winner_names = ["Red","(DRAW)","Blue"]
        double_jump_piece = (-1,-1)

        while True:
            # Get a list of legal moves
            legal_moves,any_can_jump = self.getLegalMoves(p,double_jump_piece=double_jump_piece)

            # Out of luck? End the game!
            if len(legal_moves)==0:
                winner = -p
                break
            
            if turn > max_turns:
                winner = 0
                break
            
            if not debug:
                clear_output()

            # Select one of the legal moves that the player can make
            move = legal_moves[0]
            if len(legal_moves) > 1:
                move = legal_moves[np.random.randint(0,len(legal_moves)-1)]

            # Actually execute the move
            self.performMove(move)

            # Visually plot the results of the move
            self.plot(wait_time_ms=wait_time_ms,last_move=move[1])

            # Prepare for the next loop iteration

            # If you can double jump, do it!
            i1,j1 = move[1][0],move[1][1]
            _,can_jump_again = self.getPieceMoveOptions(i1,j1)
            if move[2] == 0:
                can_jump_again = False # Can't "double jump" after a non-capture move
            if can_jump_again:
                double_jump_piece = (i1,j1)
                if debug:
                    print("#"*50)
                    print("DOUBLE JUMP DETECTED")
                    print("#"*50)
            else:
                double_jump_piece = (-1,-1)
                p = -p # switch players :)
            turn += 1
        winner_name = winner_names[winner+1]
        print(f"{winner_name} won after {turn-1} turns!")
