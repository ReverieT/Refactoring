from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Literal
from .region_manager import RegionManager, ValidSceneStatus

from vision_module.region_manager import RegionManager, ValidSceneStatus
"""
this is for 相机驱动
图像帧数据结构
"""
class ImageFrame:
    def __init__(self, color_data: np.ndarray, depth_data: np.ndarray, timestamp: float):
        self.color_data = color_data  
        self.depth_data = depth_data  
        self.timestamp = timestamp
    # def is_valid(self) -> bool:
    #     """快速验证帧数据是否有效（辅助下游判断）"""
    #     return (self.color_data is not None and self.depth_data is not None and
    #             self.color_data.size > 0 and self.depth_data.size > 0)





"""
* PackageInfo: 包裹信息，兼容之前设计
"""
# 基础设计：类型别名【TODO】:暂时先这样
PixelCoord = Tuple[int, int]  # 像素坐标 (x, y)
WorldCoord = Tuple[float, float, float]  # 世界坐标 (x, y, z)（后续对接机械臂可用）
EulerAngle = Tuple[float, float, float]  # 欧拉角 (roll, pitch, yaw)（抓取旋转角度）
RegionID = str  # 区域ID（如"left_zone"、"left_zone_sub_1768234665863"）
ArmID = str  # 机械臂ID（如"left_arm"、"right_arm"）
# 包裹状态： unsolve -> (solve) -> graspable
#                            └-> ungraspable
#                            └-> has_grasped
    # unsolve: 解算失败，多种原因需具体分析
    # ungraspable: 无法抓取，法向量问题 或 包裹尺寸问题
    # has_grasped: 已被抓取，防止重复计算
PackageStatus = Literal["unsolve", "ungraspable", "graspable", "has_grasped"]  # 包裹状态
SceneStatus = Literal["L-sort_R-sort", "L-sort_R-up", "L-up_R-up", "L-remove_R-up"]  # 场景状态
FlowState = Literal["Static", "Dynamic"]
# 基础设计：辅助类
@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
@dataclass
class Euler:
    rx: float = 0.0 # roll
    ry: float = 0.0 # pitch
    rz: float = 0.0 # yaw

@dataclass
class PackageInfo:
    # ---------------------- 1. 基础标识信息（用于追踪和溯源） ----------------------
    package_id: str  # 包裹唯一ID（如"pkg_1768234665863_001"，时间戳+序号）
    timestamp: str  # 包裹检测时间戳（与场景时间戳对齐，格式：%H-%M-%S_%f）

    # ---------------------- 2. 核心检测数据（来自视觉检测模型，原有代码核心字段） ----------------------
    obb: any  # YOLO检测结果
    center_pixel: PixelCoord  # 包裹中心像素坐标 (x, y)（用于区域判断，已转换为整数）
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
    euler_angle: EulerAngle = (0.0, 0.0, 0.0)  # 抓取欧拉角（旋转角度，原有代码中的r）
    grasp_priority: int = 1  # 抓取优先级（数值越大越优先，可基于子区域优先级调整）

    # ---------------------- 5. 生命周期状态信息（用于追踪包裹处理进度） ----------------------
    status: PackageStatus = "unsolve"  # 包裹当前状态
    error_msg: Optional[str] = None  # 异常信息（如"超出抓取范围"、"低置信度"，便于调试）

    # ---------------------- 辅助方法（可选，简化业务代码） ----------------------
    def is_graspable(self) -> bool:
        """判断包裹是否可抓取（基于区域属性和当前状态）"""
        if self.status != "detected" and self.status != "graspable":
            return False
        # 从子区域属性中获取是否允许抓取（默认允许）
        return self.sub_region_attributes.get("allow_grasp", True)

    def need_anti_collision(self) -> bool:
        """判断包裹是否需要防碰撞（基于子区域属性）"""
        return self.sub_region_attributes.get("need_anti_collision", False)

"""
场景信息说明
"""
@dataclass
class SceneInfo:
    """全局场景信息（统筹所有包裹、区域、设备状态）"""
    # ---------------------- 1. 场景元数据（唯一标识和基础配置） ----------------------
    scene_id: str  # 场景唯一ID（如"scene_1768234665863"，时间戳）
    timestamp: str  # 场景创建时间戳（与包裹时间戳对齐，格式：%H-%M-%S_%f）
    config_path: str = "./region_config.yaml"  # 区域配置文件路径（便于加载区域信息）

    # ---------------------- 2. 包裹集合信息（所有包裹的容器，与原有代码队列对应） ----------------------
    all_packages: List[PackageInfo] = field(default_factory=list)  # 所有检测到的包裹
    left_packages: List[PackageInfo] = field(default_factory=list)  # 左区包裹列表（对应left_config.parcel_list）
    right_packages: List[PackageInfo] = field(default_factory=list)  # 右区包裹列表（对应right_config.parcel_list）
    dead_zone_packages: List[PackageInfo] = field(default_factory=list)  # 死区包裹列表（无法抓取）

    # ---------------------- 3. 区域配置摘要（可选，快速获取区域信息，无需重复加载YAML） ----------------------
    region_summary: Dict = field(default_factory=dict)  # 区域配置摘要（如{"left_zone": "左区（移栽机）", ...}）

    # ---------------------- 5. 场景处理结果（用于追踪场景进度和复盘） ----------------------
    status: SceneStatus = "L-sort_R-sort"  # 场景当前状态
    process_duration: float = 0.0  # 场景处理耗时（秒）
    processed_count: int = 0  # 已处理包裹数
    grasped_count: int = 0  # 已抓取包裹数
    error_count: int = 0  # 异常包裹数

    # ---------------------- 辅助方法（可选，简化业务代码） ----------------------
    def add_package(self, package: PackageInfo) -> None:
        """添加包裹到对应集合（自动分区，简化原有代码的队列添加逻辑）"""
        self.all_packages.append(package)

        # 自动分类到左区/右区/死区
        if package.arm_id == "left_arm":
            self.left_packages.append(package)
        elif package.arm_id == "right_arm":
            self.right_packages.append(package)
        else:
            self.dead_zone_packages.append(package)
            package.status = "ungraspable"
            package.error_msg = "包裹位于死区，无关联机械臂"

    def get_graspable_count(self) -> int:
        """获取可抓取包裹总数（统计用）"""
        return len([pkg for pkg in self.all_packages if pkg.is_graspable()])

    def derive_scene_status(self, region_manager: RegionManager) -> ValidSceneStatus:
        """从区域管理器中获取左右区状态，推导全局场景状态"""
        return region_manager.derive_scene_status()


@dataclass
class VisionResult:
    """视觉模块结果封装，用于与DecisionModule交互"""
    # 基础帧数据【temp: 或许并不需要】
    region_id: str or None = None  # 区域id（"left_zone", "right_zone"）
    cmd: str or None = None  # 指令（"sort", "up", "remove"）
    has_robot: bool = False  # 是否存在机械臂遮挡
    parcel_list: List[PackageInfo] = field(default_factory=list)  # 当前场景检测到的全部包裹
    flow_result: dict or None = None  # 光流计算结果

    # 元数据（便于追溯与调试）
    timestamp: str = ""  # 处理完成时间戳
    error_msg: str = ""  # 错误信息（若处理失败）