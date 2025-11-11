import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

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
            #getPlayer normalization justified because this function is only used to see if a space is taken, and by whom
            return self.getPlayer(self.board[i][j]), abs(self.board[i][j])==2 # is it a king?
        
    def getLegalMoves(self,p):
        # p: player (+1 = blue, -1 = red)
        num_queried = 0
        n = self.n
        board = self.board
        legal_moves = [] # elements look like [start (i,j), end (i,j), move type (0 = move, 1 = capture a blue, -1 = capture a red)]

        for i in range(n):
            for j in range(n):
                current_piece, is_king = self.getPiece(i,j)
                directions = [-1,1] if is_king else [1]
                if current_piece == p:
                    num_queried += 1
                    for LR in [-1,1]: # You can go left OR right
                        for DIR in directions: # Kings can go forward OR backward
                            i1,i2,j1,j2 = i+p*DIR,i+2*p*DIR,j+LR,j+2*LR
                            
                            diag_piece,_ = self.getPiece(i1,j1)

                            if diag_piece < -1: continue # Out of bounds

                            if diag_piece == p:
                                # blocked
                                continue
                            elif diag_piece == -p:
                                # enemy! query further
                                next_diag_piece,_ = self.getPiece(i2,j2)
                                if next_diag_piece < -1: continue # Out of bounds
                                if next_diag_piece == 0:
                                    # can successfully jump! do something...
                                    legal_moves.append([(i,j),(i2,j2),-p])
                                else:
                                    # jump is blocked.
                                    pass
                            else:
                                # i've got a blank space, baby
                                legal_moves.append([(i,j),(i1,j1),0])

                    if num_queried >= self.howManyLeft(p):
                        break
            if num_queried >= self.howManyLeft(p):
                    break
        return legal_moves
        
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
