import numpy as np
import math

PASS = lambda n: n**4 - 1
EPS = 1e-8

def PASS(n): 
    return n**4 - 1

class RandomPlayer:
    def __init__(self, game, args=None):
        self.game = game
        self.args = args

    def startGame(self):
        pass

    def __call__(self, canonicalBoard):
        valids = self.game.getValidMoves(canonicalBoard, 1)
        legal = np.where(valids == 1)[0]
        if len(legal) == 0:
            return PASS(self.game.n)  # 极端兜底；正常情况下不会走到这里

        non_pass = legal[legal != PASS(self.game.n)]
        pick_from = non_pass if len(non_pass) else legal
        pick = np.random.choice(pick_from)
        return int(pick)

    def endGame(self):
        pass
