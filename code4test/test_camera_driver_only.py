import time
import cv2
import numpy as np

# 根据你的文件结构导入 CameraDriver
# 假设你的目录结构是 vision_module/camera_driver.py
from vision_module.camera_driver import CameraDriver

def test_driver_raw():
    print("========== 开始测试相机驱动 (Raw Driver) ==========")
    
    # 1. 配置参数 (与你 VisionModule 中的一致)
    SERIAL_NUMBER = '00DA5939159' # 你的真实序列号
    
    print(f"1. 正在初始化 CameraDriver (SN: {SERIAL_NUMBER})...")
    try:
        # 实例化驱动
        cam = CameraDriver(SERIAL_NUMBER, frame_queue_size=1, fetch_frame_timeout=5000)
        
        # 进入上下文 (相当于调用 connect/start)
        with cam:
            print("2. 相机启动成功，开始连续取图测试 (按 'q' 退出)...")
            
            start_time = time.time()
            frame_count = 0
            
            while True:
                # 尝试获取最新帧
                try:
                    frame = cam.get_latest_frame()
                    
                    if frame is None:
                        print("Warning: 获取到的帧为 None")
                        continue
                        
                    # 检查数据
                    # 假设 frame 对象里有 color_data (numpy array) 和 depth_data
                    color_img = frame.color_data
                    depth_img = frame.depth_data
                    
                    frame_count += 1
                    
                    # 计算实时帧率
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # 打印信息 (每30帧打印一次，避免刷屏)
                    if frame_count % 30 == 0:
                        h, w, c = color_img.shape
                        print(f"Capture OK | Size: {w}x{h} | FPS: {fps:.2f} | Depth center: {depth_img[h//2, w//2]}mm")
                    
                    # 可视化 (这一步最直观，确认画面没黑、没花)
                    cv2.putText(color_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示彩色图
                    cv2.imshow("Real Camera - Color", color_img)
                    
                    # 显示深度图 (归一化以便显示)
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imshow("Real Camera - Depth", depth_colormap)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"取图循环中发生异常: {e}")
                    break
                    
    except Exception as e:
        print(f"!!! 相机初始化或运行失败: {e}")
        print("请检查：1. USB线是否插好 (推荐USB 3.0) 2. 序列号是否正确")
    finally:
        cv2.destroyAllWindows()
        print("测试结束")

if __name__ == "__main__":
    test_driver_raw()