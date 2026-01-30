# region_manager.py
import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Literal
from pathlib import Path

# LeftRegionStatus = Literal["sort", "up", "remove"]  # 左区状态：分拣、上料、剔除
# RightRegionStatus = Literal["sort", "up"]  # 右区状态：分拣、上料
# RegionStatus = Literal["sort", "up", "remove"]  # 统一区域状态（兼容左/右区，用于BaseRegion字段）
ValidSceneStatus = Literal["L-sort_R-sort", "L-sort_R-up", "L-up_R-up", "L-remove_R-up"]

from enum import Enum

class RegionStatus(Enum):
    SORT = "sort"
    UP = "up"
    REMOVE = "remove"



# ---------------------- 数据结构（满足多边缘层+边缘向量+基座层不抓取） ----------------------
@dataclass
class SubRegion:
    """子区域类（支持多边缘层+边缘方向向量+基座层不抓取）"""
    sub_region_id: str  # 如"left_edge_1"、"right_base"
    name: str  # 如"左区边缘层1"、"右区不抓取基座层"
    type: str  # "edge"（边缘层）/ "near_base"（基座层）
    polygon_vertices: List[Tuple[int, int]]  # 凸多边形顶点（像素坐标）
    priority: int = 1  # 优先级（数值越大越高，解决重叠冲突）
    layer_index: Optional[int] = None  # 边缘层序号（1/2/3，区分同类型边缘层）
    edge_vector: Optional[Tuple[float, float]] = None  # 边缘方向向量（(dx, dy)，归一化）
    attributes: Dict = field(default_factory=lambda: {
        "need_anti_collision": False,
        "hard_to_solve": False,
        "allow_grasp": True
    })

    def __post_init__(self):
        """初始化后自动配置属性（基座层强制不抓取，边缘层强制防碰撞）"""
        if self.type == "near_base":
            self.attributes["allow_grasp"] = False
            self.attributes["hard_to_solve"] = True
        if self.type == "edge":
            self.attributes["need_anti_collision"] = True

    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """判断点是否在子区域内（凸多边形判断）"""
        if len(self.polygon_vertices) < 3:
            return False
        polygon = np.array(self.polygon_vertices, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), measureDist=False)
        return result >= 0

    def calculate_edge_vector_from_two_points(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        """从两点计算并绑定归一化边缘向量"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = np.linalg.norm([dx, dy])
        if norm > 0:
            self.edge_vector = (dx / norm, dy / norm)
        else:
            self.edge_vector = (dx, dy)

@dataclass
class BaseRegion:
    """基础区域类（左区/右区，包含多个子区域）"""
    region_id: str  # 如"left_zone"、"right_zone"
    name: str  # 如"左区（移栽机）"、"右区（传送带）"
    arm_id: str  # 如"left_arm"、"right_arm"
    polygon_vertices: List[Tuple[int, int]]  # 凸多边形顶点（像素坐标）
    sub_regions: List[SubRegion] = field(default_factory=list)
    calibration_resolution: Tuple[int, int] = (1408, 1024)  # 标定时相机分辨率

    region_status: RegionStatus = RegionStatus.SORT  # 区域当前状态（分拣、上料、剔除）


    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """判断点是否在基础区域内"""
        if len(self.polygon_vertices) < 3:
            return False
        polygon = np.array(self.polygon_vertices, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), measureDist=False)
        return result >= 0

    def get_sub_region_of_point(self, point: Tuple[int, int]) -> Optional[SubRegion]:
        """获取点所属的子区域（优先高优先级）"""
        if not self.is_point_inside(point):
            return None
        sorted_sub_regions = sorted(self.sub_regions, key=lambda x: x.priority, reverse=True)
        for sub_region in sorted_sub_regions:
            if sub_region.is_point_inside(point):
                return sub_region
        return None

# ---------------------- 区域管理器（配置加载/保存/全局判断） ----------------------
class RegionManager:
    """区域管理器（无交互，仅负责配置的加载、保存和区域判断）"""
    def __init__(self, config_path: str = "./region_config.yaml"):
        self.config_path = Path(config_path)
        self.base_regions: List[BaseRegion] = []
        # 初始化时加载现有配置（若存在）
        if self.config_path.exists():
            self.load_config()
        else:
            print(f"警告：未找到配置文件 {self.config_path}，将使用默认配置。")

    def load_config(self) -> bool:
        """从YAML文件加载区域配置"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            self.base_regions = []
            for base_data in config_data.get("base_regions", []):
                # 解析子区域
                sub_regions = []
                for sub_data in base_data.get("sub_regions", []):
                    # 优化点1：子区域顶点格式转换（list→tuple）
                    sub_vertices = [tuple(map(int, point)) for point in sub_data["polygon_vertices"]]
                    sub_region = SubRegion(
                        sub_region_id=sub_data["sub_region_id"],
                        name=sub_data["name"],
                        type=sub_data["type"],
                        polygon_vertices=sub_vertices,  # 传入转换后的顶点
                        priority=sub_data.get("priority", 1),
                        layer_index=sub_data.get("layer_index"),
                        edge_vector=sub_data.get("edge_vector")
                    )
                    sub_regions.append(sub_region)
                # 解析基础区域
                # 优化点2：基础区域顶点格式转换（list→tuple）
                base_vertices = [tuple(map(int, point)) for point in base_data["polygon_vertices"]]
                base_region = BaseRegion(
                    region_id=base_data["region_id"],
                    name=base_data["name"],
                    arm_id=base_data["arm_id"],
                    polygon_vertices=base_vertices,  # 传入转换后的顶点
                    sub_regions=sub_regions,
                    calibration_resolution=tuple(base_data.get("calibration_resolution", (1920, 1080)))
                )
                self.base_regions.append(base_region)

            print(f"成功加载配置：{self.config_path}（{len(self.base_regions)} 个基础区域）")
            return True
        except Exception as e:
            print(f"加载配置失败：{str(e)}")
            return False

    def save_config(self) -> bool:
        """将区域配置保存到YAML文件"""
        try:
            # 转换为可序列化的字典格式
            config_data = {"base_regions": []}
            for base_region in self.base_regions:
                sub_data_list = []
                for sub_region in base_region.sub_regions:
                    sub_data = {
                        "sub_region_id": sub_region.sub_region_id,
                        "name": sub_region.name,
                        "type": sub_region.type,
                        "polygon_vertices": sub_region.polygon_vertices,
                        "priority": sub_region.priority,
                        "layer_index": sub_region.layer_index,
                        "edge_vector": sub_region.edge_vector
                    }
                    sub_data_list.append(sub_data)

                base_data = {
                    "region_id": base_region.region_id,
                    "name": base_region.name,
                    "arm_id": base_region.arm_id,
                    "polygon_vertices": base_region.polygon_vertices,
                    "sub_regions": sub_data_list,
                    "calibration_resolution": base_region.calibration_resolution
                }
                config_data["base_regions"].append(base_data)

            # 保存到文件
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, indent=4, allow_unicode=True)

            print(f"成功保存配置：{self.config_path}")
            return True
        except Exception as e:
            print(f"保存配置失败：{str(e)}")
            return False

    def get_region_of_point(self, point: Tuple[int, int]) -> Tuple[Optional[BaseRegion], Optional[SubRegion]]:
        """获取点所属的基础区域和子区域（对外核心接口）"""
        for base_region in self.base_regions:
            if base_region.is_point_inside(point):
                sub_region = base_region.get_sub_region_of_point(point)
                return (base_region, sub_region)
        return (None, None)

    def get_region_status(self, region_id: str) -> RegionStatus:
        """对外接口：获取区域状态"""
        for base_region in self.base_regions:
            if base_region.region_id == region_id:
                return base_region.region_status
        return "sort"  # 默认返回分拣状态（区域不存在时）
    def set_region_status(self, region_id: str, status: RegionStatus) -> None:
        """对外接口：设置区域状态"""
        for base_region in self.base_regions:
            if base_region.region_id == region_id:
                base_region.region_status = status
                return
        print(f"警告：区域 {region_id} 不存在，无法设置状态")

    def _get_left_right_region_status(self) -> Tuple[Optional[RegionStatus], Optional[RegionStatus]]:
        """辅助方法：获取左区和右区的当前状态"""
        left_status = None
        right_status = None

        # 遍历基础区域，提取左/右区状态
        for base_region in self.base_regions:
            if base_region.region_id == "left_zone":
                left_status = base_region.region_status
            elif base_region.region_id == "right_zone":
                right_status = base_region.region_status

        # 补充默认值（防止缺少左/右区配置）
        left_status = left_status or "sort"
        right_status = right_status or "sort"

        return (left_status, right_status)

    def derive_scene_status(self) -> ValidSceneStatus:
        """核心推导方法：从左右区状态反推全局有效场景状态"""
        # 1. 获取左右区当前状态
        left_status, right_status = self._get_left_right_region_status()

        # 2. 预定义：有效场景状态映射表（用户明确的4种有效组合）
        valid_scene_mapping = {
            ("sort", "sort"): "L-sort_R-sort",
            ("sort", "up"): "L-sort_R-up",
            ("up", "up"): "L-up_R-up",
            ("remove", "up"): "L-remove_R-up"
        }

        # 3. 匹配有效组合（优先精确匹配）
        status_pair = (left_status, right_status)
        if status_pair in valid_scene_mapping:
            return valid_scene_mapping[status_pair]

        # 4. 处理无效组合（返回最接近的合法状态，容错处理）
        print(f"警告：区域状态组合 {status_pair} 无效，自动匹配最接近的合法场景状态")
        # 无效组合兜底策略（可根据业务调整）
        if left_status == "remove" and right_status != "up":
            return "L-remove_R-up"
        elif left_status == "up" and right_status == "sort":
            return "L-up_R-up"
        else:
            return "L-sort_R-sort"  # 默认兜底场景状态
"""
@@示例1@@——判断点所属区域：
region_manager = RegionManager(config_path="./region_config.yaml")
test_point = (100, 100)
base_region, sub_region = region_manager.get_region_of_point(test_point)
print(f"测试点 {test_point} 所属基础区域: {base_region.name if base_region else '死区'}")
print(f"测试点 {test_point} 所属子区域: {sub_region.name if sub_region else '无'}")
@@示例2@@——区域状态：
#获取区域状态
base_region.get_region_status()
#设置区域状态
base_region.set_region_status("up")     # base_region 为 left_zone or right_zone
"""