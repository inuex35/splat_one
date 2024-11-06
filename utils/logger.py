from loguru import logger
import os

def setup_logger(log_dir='logs', log_file='app.log'):
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, log_file))
    return logger