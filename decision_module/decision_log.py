import os 
import logging
from datetime import datetime

from utils.path_utils import get_static_path_withoutfile as static_path


def log_for_decision():
    log_dir = static_path("decision/logs")
    current_date = datetime.now().strftime("%Y%m%d")
    log_dir = os.path.join(log_dir, f"{current_date}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%H-%M-%S_%f")
    log_file_path = os.path.join(log_dir, f'{current_time}.log')
    
    logger = logging.getLogger("DecisionModule")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - [%(levelname)s] - %(filename)s:%(lineno)d - \n%(message)s',
        datefmt='%H:%M:%S'  # 仅时分秒，不包含年月日
    )
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

decision_logger = log_for_decision()