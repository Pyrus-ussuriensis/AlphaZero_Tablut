import numpy as np
import pytest
from tablut.Args import args
from tablut.rules.TaflLogic import Board
from tablut.rules.TaflGame import TaflGame as Game
from tablut.utils.Digits import int2base, base2int  # 若无就用你写的


# 获取游戏和棋盘
g = Game("Tablut")
board = g.getInitBoard()

def trans(m):
    num = -1
    for i, piece in enumerate(board.pieces):
        if piece[0] == m[0] and piece[1] == m[1]:
            num = i
    if num == -1:
        print("Wrong position num\n")
    return num
# 移动规则
## 测试移动
### 合法
@pytest.mark.parametrize("m", [[2,4,2,0],[2,4,2,2],[2,4,2,6]])
@pytest.mark.move
def test_moves1(m):
    num = trans(m)
    assert board._isLegalMove(num, m[2], m[3]) == 0

### 非法 出界，非直线，没有移动，跨过棋子
@pytest.mark.parametrize("m1", [[2,4,2,9],[2,4,2,-2],[2,4,2,16]])
@pytest.mark.parametrize("m2", [[2,4,3,7],[2,4,3,3],[2,4,3,1]])
@pytest.mark.parametrize("m3", [[2,4,2,4],[1,4,1,4],[4,4,4,4]])
@pytest.mark.parametrize("m4", [[2,4,1,4],[2,4,3,4],[2,4,0,4]])
@pytest.mark.parametrize("m5", [[2,4,6,4],[2,4,7,4],[2,4,8,4]]) # 如果跨过王座是优先检测出来
@pytest.mark.move
def test_moves2(m1, m2, m3, m4, m5):
    num = trans(m1)
    assert board._isLegalMove(num, m1[2], m1[3]) == -1 # 出界
    num = trans(m2)
    assert board._isLegalMove(num, m2[2], m2[3]) == -3 # 非直线
    num = trans(m3)
    assert board._isLegalMove(num, m3[2], m3[3]) == -4 # 没有移动
    num = trans(m4)
    assert board._isLegalMove(num, m4[2], m4[3]) == -20 # 跨过棋子
    num = trans(m5)
    assert board._isLegalMove(num, m5[2], m5[3]) == -10 # 跨过棋子

# 吃子规则
def test_killing():
    board = g.getInitBoard()
    
    num = -1
    for i, piece in enumerate(board.pieces):
        if piece[0] == 3 and piece[1] == 4:
            num = i
    move0 = [6,4,6,5]
    move1 = [3,0,3,3]
    move2 = [6,5,6,4]
    move3 = [0,5,3,5]
    board.execute_move(move0, None)
    board.execute_move(move1, None)
    board.execute_move(move2, None)
    board.execute_move(move3, None)
    assert board.pieces[num][0] == -99


# 终局 
def test_king_edge_win():
    board = g.getInitBoard()
    move0 = [5,4,5,5]
    move1 = [7,4,7,5]
    move2 = [6,4,6,5]
    move3 = [7,5,7,4]
    move4 = [4,4,6,4]
    move5 = [7,4,7,5]
    move6 = [6,4,6,0]
    board.execute_move(move0, None)
    board.execute_move(move1, None)
    board.execute_move(move2, None)
    board.execute_move(move3, None)
    board.execute_move(move4, None)
    board.execute_move(move5, None)
    board.execute_move(move6, None)
    assert board._getWinLose() == 1

def test_50move_rule():
    board = g.getInitBoard()
    board.time=args.limit
    assert board._getWinLose() == args.draw
