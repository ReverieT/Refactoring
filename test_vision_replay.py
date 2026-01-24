import threading
import time
from vision import VisionModule
from file_camera_driver import FileCameraDriver # 刚刚写的类

# 回调函数
def log_callback(msg):
    print(f"[LOG] {msg}")

def frame_callback(data):
    pass

def main():
    # 1. 指定你的图片路径
    # 请确保路径下有 color 和 depth 文件夹
    dataset_path = "./my_test_dataset" 
    
    # 2. 实例化文件驱动
    # fps设低一点，方便你观察每一帧的处理日志
    fake_cam = FileCameraDriver(data_dir=dataset_path, fps=1, loop=True)
    
    # 3. 注入 VisionModule
    # 注意：这里需要你确认 VisionModule 的 __init__ 是否已经支持传入 camera_driver
    # 如果还没有支持，请按照之前的建议修改 VisionModule 的构造函数
    cv_module = VisionModule(log_callback, frame_callback)
    
    # 【强行替换】如果你不想改 VisionModule 源码，可以用这种 Python 的动态特性强行替换
    # 但由于 VisionModule 在 __init__ 里已经调用了 camera.__enter__，
    # 所以最稳妥的方法还是修改构造函数支持注入。
    # 这里假设你已经修改了构造函数：
    # cv_module = VisionModule(log_callback, frame_callback, camera_driver=fake_cam)
    
    # 如果还没改构造函数，可以用下面的 HACK 方法（不太推荐，但在测试时有用）：
    cv_module.camera.stop_camera() # 停掉真实相机
    cv_module.camera = fake_cam    # 替换成假的
    # 手动启动假的
    cv_module.camera.__enter__()
    
    # 4. 模拟区域状态 (让 VisionModule 开始工作)
    # 强制让 region_manager 认为左侧需要分拣
    # 你可能需要 Mock 一下 RegionManager，或者简单粗暴地在 vision.py 里临时改一下
    print("开始回放测试...")
    
    # 启动主循环
    t = threading.Thread(target=cv_module.main_loop, daemon=True)
    t.start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()