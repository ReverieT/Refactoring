import threading
import time
import logging
import sys
from unittest.mock import MagicMock

# ==========================================
# 导入你的模块
# ==========================================
from vision import VisionModule
from decision import DecisionModule, ZoneIntent
from code4test.file_camera_driver import FileCameraDriver
# 确保引用的是 region_manager 里的 Enum
from vision_module.region_manager import RegionStatus

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [%(threadName)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TestIntegration")

# ==========================================
# 1. Mock 硬件层 (PostalDas) - 保持不变
# ==========================================
class MockPostalDas:
    def __init__(self):
        self.left_robot = MagicMock()
        self.right_robot = MagicMock()
        self.left_robot.packpose_flag = False
        self.right_robot.packpose_flag = False
        self.left_robot.in_camera_cover_flag = False
        
    def up_material(self, id, status):
        side = "右边(Right)" if id == 1 else "左边(Left)" if id == 2 else "未知"
        action = "开始上料 [START]" if status else "停止上料 [STOP]"
        # 仅在状态改变时打印，防止刷屏
        logger.warning(f"🤖 [硬件] {side} {action}")

    def remove_material(self, status):
        action = "开始剔除 [START]" if status else "停止剔除 [STOP]"
        logger.warning(f"🤖 [硬件] {action}")

    def isStart(self):
        return True

# ==========================================
# 2. 辅助回调
# ==========================================
def vision_frame_callback(data):
    pass
def vision_log_callback(msg):
    pass

# ==========================================
# 3. 主测试逻辑
# ==========================================
def main():
    print("========== 启动 Vision & Decision 真实闭环测试 ==========")
    
    # --- A. 准备模拟相机 ---
    dataset_path = "./test_data"
    print(f"1. 加载虚拟相机: {dataset_path}")
    fake_cam = FileCameraDriver(data_dir=dataset_path, fps=1.0, loop=True, specific_index=None)

    # --- B. 初始化模块 ---
    print("2. 初始化模块...")
    vision_module = VisionModule(vision_log_callback, vision_frame_callback)
    
    if hasattr(vision_module, 'camera'):
        try: vision_module.camera.stop_camera() 
        except: pass
    vision_module.camera = fake_cam
    vision_module.camera.__enter__()

    # --- C. 兼容性修补 (Monkey Patch) ---
    # [关键修复]：你的 decision.py 调用了 self.vision_module.get_region_status
    # 但原始 vision.py 可能没有在这个类上定义该方法，而是定义在 vision_module.region_manager 上。
    # 为了防止报错，我们在测试中动态给 vision_module 加上这两个转发方法。
    
    if not hasattr(vision_module, 'get_region_status'):
        print("🔧 [自动修补] 给 VisionModule 添加 get_region_status 转发方法")
        vision_module.get_region_status = lambda region_id: vision_module.region_manager.get_region_status(region_id)
        
    if not hasattr(vision_module, 'set_region_status'):
        print("🔧 [自动修补] 给 VisionModule 添加 set_region_status 转发方法")
        vision_module.set_region_status = lambda region_id, status: vision_module.region_manager.set_region_status(region_id, status)

    # 实例化 Decision
    mock_das = MockPostalDas()
    decision_module = DecisionModule(vision_module, mock_das)

    # --- D. [重要改变] 不再劫持 RegionManager ---
    # 旧测试中的 dynamic_get_status 代码已被删除。
    # 现在完全依赖 decision.py 中的 set_left_zone_state -> vision_module.set_region_status -> RegionManager
    # 实现了真正的代码逻辑闭环验证。
    print("3. 使用真实闭环逻辑 (Decision -> RegionManager -> Vision)")

    # --- E. 启动线程 ---
    print("4. 启动双线程...")
    t_vision = threading.Thread(target=vision_module.main_loop, name="VisionThread", daemon=True)
    t_vision.start()

    t_decision = threading.Thread(target=decision_module.make_decision, name="DecisionThread", daemon=True)
    t_decision.start()

    # --- F. 监控主循环 ---
    print("========== 系统运行中 (按 Ctrl+C 停止) ==========")
    try:
        while True:
            time.sleep(1)
            
            # 1. 获取 Decision 认为的状态
            dec_l_state = decision_module.left_zone_state
            
            # 2. 获取 RegionManager (Vision端) 实际存储的状态
            #    如果闭环成功，这两个值应该在短暂延迟后保持一致
            vis_l_state = vision_module.region_manager.get_region_status('left_zone')
            
            # 3. 打印对比
            match_icon = "✅" if dec_l_state.name == vis_l_state.name.upper() else "⏳" # Enum vs Enum/String check
            
            # 获取队列长度
            l_q = decision_module.left_catch_queue.qsize()
            
            print(f"\r[闭环监控] 决策意图:{dec_l_state.name:<4} -> {match_icon} -> 视觉状态:{vis_l_state.name:<4} | 队列:{l_q} | 策略:{decision_module.current_state}", end="")
            
            # 模拟机械臂取货
            if not decision_module.left_catch_queue.empty():
                decision_module.left_catch_queue.get()
                # logger.info("抓取...") # 减少日志干扰

    except KeyboardInterrupt:
        print("\n\n测试停止...")

if __name__ == "__main__":
    main()