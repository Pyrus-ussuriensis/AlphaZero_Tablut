
# Note: Run this file from Arena directory (the one above /tafl)

from tablut import Arena
from tablut.models.MCTS import MCTS
from tablut.rules.TaflGame import TaflGame, display
from tablut.baselines.TaflPlayers import *
from tablut.models.Players import MCTSPlayer
from tablut.models.NNet import NNetWrapper as nn
from tablut.Args import args
from tablut.baselines.Elo_Cal import Evaluate_Model_with_Alpha_Beta
#from tafl.keras.NNet import NNetWrapper as NNet

import numpy as np
from tablut.utils.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TaflGame("Tablut")

# all players
rp = RandomTaflPlayer(g).play
gp = GreedyTaflPlayer(g).play
hp = HumanTaflPlayer(g).play

from tablut.baselines.alphabeta_player import AlphaBetaTaflPlayer
from tablut.baselines.greedy_player import GreedyTaflPlayer
from tablut.baselines.random_player import RandomPlayer

a, b, c = AlphaBetaTaflPlayer(g,2), GreedyTaflPlayer(g), RandomPlayer(g)

nnet = nn(g)
pnet = nn(g)
checkpoint = "./docs/"
nnet.load_checkpoint(checkpoint, args.load_folder_file)
pnet.load_checkpoint(checkpoint, args.load_folder_file)
pmcts_player = MCTSPlayer(g, pnet, args, temp=0)
nmcts_player = MCTSPlayer(g, nnet, args, temp=0)

#arena = Arena.Arena(pmcts_player, nmcts_player, g)
#print(arena.playGames(64, verbose=False))

print(Evaluate_Model_with_Alpha_Beta(new_model=nmcts_player, g=g, n=2, d=2))
#print(Evaluate_Model_with_Alpha_Beta(new_model=a, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=b, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=c, g=g))
#print(Evaluate_Model_with_Alpha_Beta(new_model=a, g=g, n=6))