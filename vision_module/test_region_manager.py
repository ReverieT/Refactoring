# 新建test_region.py（仅用于测试，可选）
from region_manager import RegionManager

def test_yaml_loading():
    """测试加载YAML配置文件"""
    # 1. 初始化区域管理器，自动加载当前目录下的region_config.yaml
    region_manager = RegionManager(config_path="./region_config.yaml")

    # 2. 验证加载结果（打印基础区域和子区域信息）
    for base_region in region_manager.base_regions:
        print(f"\n=== 基础区域：{base_region.name}（ID：{base_region.region_id}）===")
        print(f"  关联机械臂：{base_region.arm_id}")
        print(f"  标定分辨率：{base_region.calibration_resolution}")
        print(f"  包含子区域数：{len(base_region.sub_regions)}")

        for sub_region in base_region.sub_regions:
            print(f"  - 子区域：{sub_region.name}（类型：{sub_region.type}）")

    # 3. 测试点是否在区域内（任选一个像素坐标测试，比如左区内部点(200, 300)）
    test_point1 = (200, 300)  # 左区内部点
    test_point2 = (800, 300)  # 右区内部点
    test_point3 = (0, 0)      # 死区点

    base_region1, sub_region1 = region_manager.get_region_of_point(test_point1)
    base_region2, sub_region2 = region_manager.get_region_of_point(test_point2)
    base_region3, sub_region3 = region_manager.get_region_of_point(test_point3)

    print(f"\n=== 区域判断测试 ===")
    print(f"  点{test_point1}：所属区域={base_region1.name if base_region1 else '死区'}，所属子区域={sub_region1.name if sub_region1 else '无'}")
    print(f"  点{test_point2}：所属区域={base_region2.name if base_region2 else '死区'}，所属子区域={sub_region2.name if sub_region2 else '无'}")
    print(f"  点{test_point3}：所属区域={base_region3.name if base_region3 else '死区'}，所属子区域={sub_region3.name if sub_region3 else '无'}")

if __name__ == "__main__":
    test_yaml_loading()