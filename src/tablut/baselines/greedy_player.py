import numpy as np

PASS = lambda n: n**4 - 1
# 公共启发式对当前方分数越大越好
def _eval_board(game, b) -> float:
    img = b.getImage()
    n = len(img)

    # 找王：abs(t)==2 t>0 说明我方是守方 t<0 说明我方是攻方
    k = next(((x, y, t) for x, y, t in b.pieces if abs(t) == 2), None)
    if k is None:
        # 此时应已由 getGameEnded 捕获，防御返回极端小值
        return -1e9
    x, y, tking = k
    sgn_king = 1 if tking > 0 else -1   # 守方(+1)希望近边/高机动；攻方(-1)相反

    # 到边距离与机动性
    dist_edge = min(x, y, n-1-x, n-1-y)
    mob = 0
    occ = {(px, py) for px, py, _ in b.pieces}
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        cx, cy = x, y
        while True:
            nx, ny = cx+dx, cy+dy
            if nx < 0 or ny < 0 or nx >= n or ny >= n: break
            if (nx, ny) in occ: break
            mob += 1; cx, cy = nx, ny

    # 子力差：我方为正
    my_cnt  = sum(1 for _, _, t in b.pieces if t > 0)
    opp_cnt = sum(1 for _, _, t in b.pieces if t < 0)

    # 王相关项按王符号定向：守方(+1)→ -dist_edge + mob；攻方(-1)→ +dist_edge - mob
    king_term = (-1.5 * dist_edge + 0.2 * mob) * sgn_king
    mat_term  =  0.05 * (my_cnt - opp_cnt)

    return king_term + mat_term


# 一层贪心
class GreedyTaflPlayer:
    def __init__(self, game):
        self.game = game

    def startGame(self):
        pass

    def endGame(self):  
        pass

    def __call__(self, canonicalBoard):
        cur = canonicalBoard.getPlayerToMove() if hasattr(canonicalBoard, "getPlayerToMove") else 1
        n = self.game.n
        valids = self.game.getValidMoves(canonicalBoard, cur)
        legal = np.flatnonzero(valids == 1)
        if len(legal) == 0:
            return None

        best, best_as = -1e18, []
        for a in legal:
            nb, _ = self.game.getNextState(canonicalBoard, cur, int(a))
            nb_me = self.game.getCanonicalForm(nb, cur)
            sc = _eval_board(self.game, nb_me)
            if sc > best + 1e-9:
                best, best_as = sc, [a]
            elif abs(sc - best) <= 1e-9:
                best_as.append(a)

        return int(np.random.choice(best_as)) if best_as else int(legal[0])

