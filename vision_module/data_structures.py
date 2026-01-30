import numpy as np
from enum import Enum, auto
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
"""
this is for 相机驱动
图像帧数据结构
"""
class ImageFrame:
    def __init__(self, color_data: np.ndarray, depth_data: np.ndarray, timestamp: float):
        self.color_data = color_data  
        self.depth_data = depth_data  
        self.timestamp = timestamp

# SceneStatus = Literal["L-sort_R-sort", "L-sort_R-up", "L-up_R-up", "L-remove_R-up"]  # 场景状态
# FlowState = Literal["Static", "Dynamic"]
# TODO 检查状态是否回环

# ==========================================
# Enums (状态定义)
# ==========================================

class FlowState(Enum):
    Static = auto()
    Dynamic = auto()

# 包裹状态： unsolve -> (solve) -> graspable
#                            └-> ungraspable
#                            └-> has_grasped
        # unsolve: 解算失败，多种原因需具体分析
        # ungraspable: 无法抓取，法向量问题 或 包裹尺寸问题
        # has_grasped: 已被抓取，防止重复计算
class PackageStatus(Enum):
    UNSOLVE = auto()      # 未解算
    SOLVE = auto()        # 已解算
    UNGRASPABLE = auto()  # 不可抓取
    GRASPABLE = auto()    # 可抓取
    HAS_GRASPED = auto()  # 已抓取

class SceneStatus(Enum):
    L_sort_R_sort = auto()
    L_sort_R_up = auto()
    L_up_R_up = auto()
    L_remove_R_up = auto()

# ==========================================
# 基础数据结构
# ==========================================
# 基础设计：类型别名【TODO】:暂时先这样
PixelCoord = Tuple[int, int]  # 像素坐标 (x, y)
RegionID = str  # 区域ID（如"left_zone"、"left_zone_sub_1768234665863"）
# TODO int or str
ArmID = int  # 机械臂ID（如"left_arm"、"right_arm"）

@dataclass
class ImageFrame:
    """相机原始帧数据"""
    color_data: np.ndarray
    depth_data: np.ndarray
    timestamp: str  
@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
@dataclass
class EulerAngle:
    rx: float = 0.0 # roll
    ry: float = 0.0 # pitch
    rz: float = 0.0 # yaw

# ==========================================
# 核心业务对象
# ==========================================

"""
PackageInfo: 包裹信息，兼容ParcelInfo
"""
@dataclass
class PackageInfo:
    # ---------------------- 1. 基础标识信息（用于追踪和溯源） ----------------------
    package_id: Optional[str] = None # 包裹唯一ID（如"pkg_1768234665863_001"，时间戳+序号）
    timestamp: Optional[str] = None  # 包裹检测时间戳（与场景时间戳对齐，格式：%H-%M-%S_%f）

    # ---------------------- 2. 核心检测数据（来自视觉检测模型，原有代码核心字段） ----------------------
    obb: Optional[Any] = None  # YOLO检测结果
    center_pixel: Optional[PixelCoord] = None # 包裹中心像素坐标 (x, y)（用于区域判断，已转换为整数）
    width: float = 0.0  # 包裹2D宽度（像素/毫米，可选）
    height: float = 0.0  # 包裹2D长度（像素/毫米，可选）
    obb_info: Dict = field(default_factory=dict)  # 完整OBB信息（如xywhr、confidence等）
        # {
        # 'rect3d':   rect3d,       
        # 'center_v': center3d_v,   相机坐标系，包裹中心点  -> 包裹高度：1100 - center3d_v[2]
        # 'center':   center3d,     相机坐标系，机械臂末端中心点（和包裹中心点相隔吸盘距离）
        # 'R'     :   R_mat,        
        # 'u_axis':   u_rot,        长轴方向
        # 'v_axis':   v_rot,        短轴方向
        # 'normal':   n,            法线方向
        # 'stay3d':   stay3d,       相机坐标系，机械臂末端停留点
        # 'long_edge': long_edge,   长边长度
        # 'short_edge': short_edge  短边长度
        # }

    # ---------------------- 3. 区域归属信息（来自region_config.yaml，区域标定结果） ----------------------
    region_id: Optional[RegionID] = None  # 所属基础区域ID（如"left_zone"、"right_zone"）
    region_name: Optional[str] = None  # 所属基础区域名称（如"左区（移栽机）"）
    sub_region_id: Optional[RegionID] = None  # 所属子区域ID（如"left_zone_sub_1768234665863"）
    sub_region_name: Optional[str] = None  # 所属子区域名称（如"left_area_top"）
    arm_id: Optional[ArmID] = None  # 关联机械臂ID（如"left_arm"，从基础区域获取）
    sub_region_attributes: Dict = field(default_factory=dict)  # 子区域属性（如allow_grasp、need_anti_collision）
    edge_vector: Optional[Tuple[float, float]] = None  # 所属边缘层向量（仅子区域类型为edge/custom有效）

    # ---------------------- 4. 抓取规划信息（用于机械臂执行，原有代码核心字段） ----------------------
    grasp_point: Position = field(default_factory=Position)  # 抓取点坐标（世界/像素）
    stay_point: Position = field(default_factory=Position)  # 停留点坐标（世界/像素）
    up_point: Position = field(default_factory=Position)  # 上升点坐标（世界/像素）
    euler_angle: EulerAngle = field(default_factory=EulerAngle) # 抓取欧拉角（旋转角度，原有代码中的r）
    grasp_priority: int = 1  # 抓取优先级（数值越大越优先，可基于子区域优先级调整）

    # ---------------------- 5. 生命周期状态信息（用于追踪包裹处理进度） ----------------------
    status: PackageStatus = PackageStatus.UNSOLVE  # 包裹当前状态
    error_msg: Optional[str] = None  # 异常信息（如"超出抓取范围"、"低置信度"，便于调试）

    # ---------------------- 辅助方法（可选，简化业务代码） ----------------------
    def is_graspable(self) -> bool:
        """判断包裹是否可抓取（基于区域属性和当前状态）"""
        if self.status != PackageStatus.GRASPABLE:
            return False
        # 从子区域属性中获取是否允许抓取（默认允许）
        return self.sub_region_attributes.get("allow_grasp", True)

    def need_anti_collision(self) -> bool:
        """判断包裹是否需要防碰撞（基于子区域属性）"""
        return self.sub_region_attributes.get("need_anti_collision", False)

@dataclass
class VisionResult:
    """视觉模块结果封装，用于与DecisionModule交互"""
    # 基础帧数据【temp: 或许并不需要】
    region_id: Optional[str] = None  # 区域id（"left_zone", "right_zone"）
    cmd: Optional[str] = None  # 指令（"sort", "up", "remove"）
    has_robot: bool = False  # 是否存在机械臂遮挡
    parcel_list: List[PackageInfo] = field(default_factory=list)  # 当前场景检测到的全部包裹

    # 元数据（便于追溯与调试）
    timestamp: str = ""  # 处理完成时间戳
    error_msg: str = ""  # 错误信息（若处理失败）