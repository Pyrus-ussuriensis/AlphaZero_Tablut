import logging
import os, torch
from logging.handlers import RotatingFileHandler
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter
from tablut.Args import *

def init_logging():
    # 建立Tensorboard的写对象
    if args.load_model:
        meta = torch.load(os.path.join(args.checkpoint, "resume.pt"), map_location="cpu")
        writer = SummaryWriter(log_dir=meta["writer_path"], purge_step=meta["i"])
    else:
        writer = SummaryWriter(log_dir='tensorboard/'+f'experiment{args.experiment}')
    # 建立日志记录对象
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 文件最大 10MB，保留 5 个备份
    handler = RotatingFileHandler(f'logs/experiment{args.experiment}.log', maxBytes=10*1024*1024, backupCount=5)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    # 记录参数的配置
    logger.info("current configuration\n%s", pformat(args))
    logger.info('start training')
    return logger, writer

logger, writer = init_logging()

