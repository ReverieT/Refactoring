import os
import logging
from datetime import datetime
from utils import path_utils

def log_for_CV(module_name, logfile_name):
    # log_dir = os.path.expanduser('./static/vision/logs')
    log_dir = path_utils.get_static_path_withoutfile("vision/logs")
    current_date = datetime.now().strftime("%Y%m%d")
    log_dir = os.path.join(log_dir, f"{current_date}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_date = datetime.now().strftime("%H-%M-%S_%f")
    log_file_path = os.path.join(log_dir, f'{current_date}.log')
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(  
    #     '%(asctime)s.%(msecs)03d - [%(levelname)s] - %(filename)s:%(lineno)d - \n%(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - [%(levelname)s] - %(filename)s:%(lineno)d - \n%(message)s',
        datefmt='%H:%M:%S'  # 仅时分秒，不包含年月日
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

Vision_logger = log_for_CV(__name__,"CVModule")

if __name__ == "__main__":
    print(__name__)