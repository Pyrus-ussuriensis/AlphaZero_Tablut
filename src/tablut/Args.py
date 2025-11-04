from tablut.utils.utils import *


args = dotdict({
    # 迭代与数据
    'numIters': 30,                       # 总轮数
    'numEps': 160,                          # 每轮自博弈局数（首轮样本稳过 batch）#####1
    'numItersForTrainExamplesHistory': 9, # 保留最近9轮构成经验池
    'step': 1000,
    'batch_size': 64,
    'maxlenOfQueue': 200_000,              # 经验窗口
    # MCTS
    'numMCTSSims': 200,    # 提升π质量#####2
    'cpuct_c_base': 19652,
    'cpuct_c_init': 1.25,
    ## 根注噪
    "dirichlet_alpha_sum": 10.0,
    'dirichlet_alpha': 0.30,
    'noise_eps': 0.25,
    # 评测/门控
    'arenaCompare': 64,    # 偶数，换边#####3
    'evaluate': 32,       # 确认赛#####4
    'updateThreshold': 0.54,
    'updateLow': 0.52,
    'updateHigh': 0.60,

    # 存档
    'checkpoint': './experiment/7/',#####5
    #'load_model': True,#####6
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 7,#####7
    # 和棋配置
    'limit': 100,
    'limit_progress': 60,
    'draw': -1e-6,
    # alphabeta老师数据
    'ab_data': 0,

})
