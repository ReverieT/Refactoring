"""
此文件设置视觉工具箱，包含工具函数
"""
####################
# 系统相关
# 1. 系统路径设置相关
####################
import os
import sys
base_path = "D:\PostalDAS"
def get_vision_resource_path(file_name: str):
    """获取视觉资源文件夹路径
    资源文件包括：|目标检测模型|配置文件|
    """
    full_path = os.path.join(base_path, "resources", "vision", file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"资源文件 {file_name} 不存在于路径 {full_path}")
    return full_path
def get_vision_static_path(file_name: str):
    """获取视觉静态文件夹路径
    静态文件包括：|图片|日志|
    """
    full_path = os.path.join(base_path, "static", "vision", file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"静态文件 {file_name} 不存在于路径 {full_path}")
    return full_path