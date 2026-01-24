import os
import cv2
import time
import numpy as np
import logging
from vision_module.data_structures import ImageFrame

# 复用原来的 ImageFrame 定义
# 假设你的目录结构允许这样导入，或者把这个类放在同级目录
# from vision_module.data_structures import ImageFrame 

logger = logging.getLogger(__name__)

class FileCameraDriver:
    """
    文件模拟相机驱动
    用于读取本地数据集，模拟相机取流过程
    """
    def __init__(self, data_dir, fps=5, loop=True):
        self.data_dir = data_dir
        self.color_dir = os.path.join(data_dir, "color")
        self.depth_dir = os.path.join(data_dir, "depth")
        self.fps = fps
        self.loop = loop # 是否循环播放
        
        # 获取文件列表并排序
        self.files = sorted([f for f in os.listdir(self.color_dir) if f.endswith(('.jpg', '.png'))])
        self.total_frames = len(self.files)
        self.current_index = 0
        self.last_fetch_time = 0
        
        if self.total_frames == 0:
            raise FileNotFoundError(f"在 {self.color_dir} 中未找到图片")
            
        logger.info(f"FileCameraDriver 初始化完成，共加载 {self.total_frames} 帧数据")

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
            # logger.warning(f"缺少深度图 {depth_path}，使用虚拟深度")
            h, w = color_img.shape[:2]
            depth_img = np.ones((h, w), dtype=np.uint16) * 1000 

        # 4. 封装成 ImageFrame
        # 这里的 timestamp 可以用当前时间，也可以解析文件名里的时间
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        
        frame = ImageFrame(
            color_data=color_img,
            depth_data=depth_img,
            timestamp=timestamp
        )

        # 指针后移
        self.current_index += 1
        
        return frame