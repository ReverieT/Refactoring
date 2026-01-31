import threading
import time
import sys
from unittest.mock import MagicMock

# 导入你的模块
from vision import VisionModule
from file_camera_driver import FileCameraDriver
# 导入枚举，用于Mock状态
from vision_module.region_manager import RegionStatus


# 【修改点1】更新回调函数，直接解包结果
def frame_callback(data):
    # 根据 VisionModule 的修改，传出来的 data 是 (left_result, right_result)
    left_res, right_res = data
    print(f"\n[CALLBACK] 收到决策数据 -> 左区包裹数: {len(left_res)} | 右区包裹数: {len(right_res)}")

# 用于过滤 VisionModule 内部 logger 的回调（如果你保留了这个接口的话）
def log_callback(msg):
    pass

def main():
    print("========== 启动单帧静态场景测试 (锁定单张图片循环) ==========")

    dataset_path = "./test_data" 
    
    # 【修改点2】使用特定索引 + 循环播放
    # specific_index=0 表示只测试文件夹里的第1张图
    # loop=True 表示无限循环处理这张图，方便你看日志报表
    print("1. 加载虚拟相机驱动 (锁定第 0 张图)...")
    fake_cam = FileCameraDriver(
        data_dir=dataset_path, 
        fps=0.5,             # 2秒处理一次，给你留时间看日志
        loop=True,           # 无限循环
        specific_index=None     # 【请修改】这里填你想测试的那张图的索引 (0, 1, 2...)
    )
    
    print("2. 初始化视觉模块...")
    try:
        # 初始化模块
        cv_module = VisionModule(log_callback, frame_callback)
    except Exception as e:
        print(f"初始化异常 (可能是真实相机连接失败): {e}")
        # 如果这里崩了，说明 __init__ 里没做好异常处理，需要去 vision.py 调整
        return

    # 3. 注入虚拟相机
    print("3. 注入虚拟相机...")
    if hasattr(cv_module, 'camera'):
        try:
            cv_module.camera.stop_camera()
        except: pass
    cv_module.camera = fake_cam
    cv_module.camera.__enter__() # 启动虚拟相机

    # 4. Mock RegionManager (强制左右区均为 SORT)
    print("4. Mock 区域状态为 [L: SORT, R: SORT]...")
    def mock_get_status(region_id):
        return RegionStatus.SORT
    
    cv_module.region_manager.get_region_status = mock_get_status
    
    # 5. 启动主循环
    print("5. 启动主线程...")
    t = threading.Thread(target=cv_module.main_loop, daemon=True)
    t.start()

    # 6. 监控测试进度
    # 【修改点3】由于现在是串行阻塞的，没有队列可以监控了
    # 我们只需要让主线程活着，静静地看控制台输出的“报表”即可
    try:
        while True:
            time.sleep(1)
            # 你可以在这里打印一点心跳，证明测试脚本没死
            print("... 测试运行中 ...")
            
    except KeyboardInterrupt:
        print("\n测试手动停止")
    
    print("========== 测试结束 ==========")

if __name__ == "__main__":
    main()