import os
import cv2
import time
import numpy as np
import logging
from vision_module.data_structures import ImageFrame

logger = logging.getLogger(__name__)

class FileCameraDriver:
    """
    文件模拟相机驱动
    用于读取本地数据集，模拟相机取流过程
    """
    # 【修改点 1】增加 specific_index 参数，默认 None
    def __init__(self, data_dir, fps=5, loop=True, specific_index=None):
        self.data_dir = data_dir
        self.color_dir = os.path.join(data_dir, "color")
        self.depth_dir = os.path.join(data_dir, "depth")
        self.fps = fps
        self.loop = loop 
        
        # 1. 获取所有文件并排序
        all_files = sorted([f for f in os.listdir(self.color_dir) if f.endswith(('.jpg', '.png'))])
        
        if not all_files:
            raise FileNotFoundError(f"在 {self.color_dir} 中未找到图片")

        # 【修改点 2】如果是指定索引模式，只加载那一张图
        if specific_index is not None:
            if 0 <= specific_index < len(all_files):
                target_file = all_files[specific_index]
                logger.info(f"== 锁定测试模式 == 仅使用第 {specific_index} 张图片: {target_file}")
                self.files = [target_file] # 列表里只留这一张图
            else:
                raise IndexError(f"索引 {specific_index} 超出范围，文件夹中共有 {len(all_files)} 张图")
        else:
            # 正常模式，加载所有图
            self.files = all_files
            logger.info(f"FileCameraDriver 初始化完成，共加载 {len(self.files)} 帧数据")

        self.total_frames = len(self.files)
        self.current_index = 0
        self.last_fetch_time = 0

    def __enter__(self):
        logger.info("FileCamera 启动")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("FileCamera 关闭")

    def get_latest_frame(self):
        """模拟获取最新帧"""
        # 1. 模拟帧率控制
        current_time = time.time()
        wait_time = (1.0 / self.fps) - (current_time - self.last_fetch_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_fetch_time = time.time()

        # 2. 检查索引
        if self.current_index >= self.total_frames:
            if self.loop:
                self.current_index = 0
                # 如果是单张图循环，不打印日志，避免刷屏
                if self.total_frames > 1:
                    logger.info("播放结束，重新开始循环")
            else:
                return None # 播放结束

        # 3. 读取文件
        filename = self.files[self.current_index]
        color_path = os.path.join(self.color_dir, filename)
        # 假设深度图文件名与彩色图一致，只是后缀可能是png
        depth_filename = filename.replace('.jpg', '.png') 
        depth_path = os.path.join(self.depth_dir, depth_filename)

        # 读取彩色图
        color_img = cv2.imread(color_path)
        
        # 读取深度图 (注意 flag=-1 读取原始16位深度)
        if os.path.exists(depth_path):
            depth_img = cv2.imread(depth_path, -1)
        else:
            # 如果没有深度图，造一个假的（假设距离相机1米）
            h, w = color_img.shape[:2]
            depth_img = np.ones((h, w), dtype=np.uint16) * 1000 

        # 4. 封装成 ImageFrame
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        
        frame = ImageFrame(
            color_data=color_img,
            depth_data=depth_img,
            timestamp=timestamp
        )

        # 指针后移
        self.current_index += 1
        
        return frame