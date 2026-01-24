# vision_module/camera_driver.py
from Mv3dRgbdImport.Mv3dRgbdDefine import *
from Mv3dRgbdImport.Mv3dRgbdApi import *
from Mv3dRgbdImport.Mv3dRgbdDefine import CoordinateType_RGB

import numpy as np
import cv2
import queue
import threading
from datetime import datetime

from ctypes import pointer, byref, c_uint
import logging

# 配置接入前，暂时使用本文件的日志【TODO: 后续接入日志系统】
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from .data_structures import ImageFrame

class CameraDriver:
    """相机驱动类，封装Mv3dRgbd SDK的初始化、采集、资源释放全流程"""
    def __init__(
        self, 
        serial_number: str = '00DA5939159',
        frame_queue_size: int = 1,
        fetch_frame_timeout: int = 5000
    ):
        # 属性集
        self.serial_number = serial_number
        self.intrinsics = {}
        # 资源管理
        self.camera_handle = None  
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.capture_thread = None
        self.stop_event = threading.Event()  # 标志：线程停止信号（替代单纯布尔值）
        # 调控参数
        self.fetch_frame_timeout = fetch_frame_timeout  
        # 状态/标志
        self.is_initialized = False
        self.is_opened = False
        self.is_streaming = False

    def initialize(self):
        """初始化相机SDK"""
        try:
            Mv3dRgbd.MV3D_RGBD_Initialize()
            self.is_initialized = True
            logger.info(f"相机SDK初始化完成")
        except Exception as e:
            logger.error(f"相机SDK初始化失败：{str(e)}", exc_info=True)
            raise

    def open_camera(self):
        """打开相机设备并获取内参"""
        if not self.is_initialized:
            raise Exception(f"相机SDK未初始化，无法打开相机")
        
        try:
            self.camera_handle = Mv3dRgbd()
            self.camera_handle.MV3D_RGBD_OpenDeviceBySerialNumber(self.serial_number)
            
            # 获取相机内参（保留stCalibInfo写法，符合你的要求）
            stCalibInfo = MV3D_RGBD_CALIB_INFO()
            self.camera_handle.MV3D_RGBD_GetCalibInfo(CoordinateType_RGB, pointer(stCalibInfo))
            self.intrinsics['fx'] = stCalibInfo.stIntrinsic.fData[0]
            self.intrinsics['fy'] = stCalibInfo.stIntrinsic.fData[4]
            self.intrinsics['cx'] = stCalibInfo.stIntrinsic.fData[2]
            self.intrinsics['cy'] = stCalibInfo.stIntrinsic.fData[5]
            
            self.is_opened = True
            logger.info(f"相机 {self.serial_number} 打开完成，内参已获取")
        except Exception as e:
            logger.error(f"相机 {self.serial_number} 打开失败：{str(e)}", exc_info=True)
            # 兜底释放已创建的句柄
            if self.camera_handle is not None:
                try:
                    self.camera_handle.MV3D_RGBD_CloseDevice()
                except:
                    pass
            self.camera_handle = None
            raise

    def start_camera(self):
        """启动相机取流"""
        if not self.is_opened:
            raise Exception(f"相机 {self.serial_number} 未打开，无法启动取流")
        
        try:
            self.camera_handle.MV3D_RGBD_Start()
            self.is_streaming = True
            self.stop_event.clear()  # 重置停止信号（确保线程可以正常运行）
            logger.info(f"相机 {self.serial_number} 取流开始")
        except Exception as e:
            logger.error(f"相机 {self.serial_number} 启动取流失败：{str(e)}", exc_info=True)
            raise

    def stop_camera(self):
        """停止相机取流并清空队列"""
        if self.is_streaming:
            self.stop_event.set()  # 发送停止信号，通知采集线程退出
            self.camera_handle.MV3D_RGBD_Stop()
            self.is_streaming = False
            
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()  # 【HACK】可能这里的代码是有问题的
            logger.info(f"相机 {self.serial_number} 取流已停止，帧队列已清空")
        else:
            logger.info(f"相机 {self.serial_number} 未处于取流状态，无需停止")

    def close_camera(self):
        """关闭相机设备"""
        if self.is_opened:
            self.camera_handle.MV3D_RGBD_CloseDevice()
            self.is_opened = False
            logger.info(f"相机 {self.serial_number} 已关闭")
        else:
            logger.info(f"相机 {self.serial_number} 未打开，无需关闭")

    def release(self):
        """释放相机SDK资源"""
        if self.is_initialized:
            Mv3dRgbd.MV3D_RGBD_Release()
            self.is_initialized = False
            logger.info(f"相机SDK已释放")
        else:
            logger.info(f"相机SDK未初始化，无需释放")

    def _capture_frames(self):
        """相机采集核心逻辑（线程执行函数）"""
        logger.info(f"相机 {self.serial_number} 采集线程已启动")
        while not self.stop_event.is_set() and self.is_streaming:
            # 初始化帧数据结构（保留stFrameData写法，符合你的要求）
            stFrameData = MV3D_RGBD_FRAME_DATA()
            color_data = None
            depth_data = None

            try:
                # 从相机获取帧数据
                self.camera_handle.MV3D_RGBD_FetchFrame(pointer(stFrameData), self.fetch_frame_timeout)

                # 解析帧数据（深度图+彩色图）
                for i in range(0, stFrameData.nImageCount):
                    width = stFrameData.stImageData[i].nWidth
                    height = stFrameData.stImageData[i].nHeight
                    np_array = np.ctypeslib.as_array(
                        stFrameData.stImageData[i].pData,
                        shape=(stFrameData.stImageData[i].nDataLen,)
                    )

                    if i == 0:  # 深度图（uint16格式）
                        depth_data = np.frombuffer(np_array, dtype=np.uint16).reshape(height, width)
                    elif i == 1:  # 彩色图（RGB转BGR，适配cv2）
                        planar = np_array.reshape((3, height, width))
                        color_data = np.transpose(planar, (1, 2, 0))
                        color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

                # 帧数据有效性验证（核心优化：无效帧直接丢弃）
                if self._validate_frame_data(color_data, depth_data):
                    # 封装为ImageFrame数据结构，添加时间戳
                    frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    image_frame = ImageFrame(
                        color_data=color_data,
                        depth_data=depth_data,
                        timestamp=frame_timestamp
                    )

                    # 非阻塞放入队列，满时丢弃旧帧（核心优化：保证线程安全+最新帧）
                    self._put_frame_to_queue(image_frame)
                else:
                    logger.warning(f"相机 {self.serial_number} 采集到无效帧，已丢弃")

            except ValueError as e:
                logger.debug(f"相机 {self.serial_number} 取图数据解析失败，重试中：{str(e)}")
                continue
            except Exception as e:
                logger.error(f"相机 {self.serial_number} 采集帧异常：{str(e)}", exc_info=True)
                continue

    def _validate_frame_data(self, color_data: np.ndarray, depth_data: np.ndarray) -> bool:
        """验证帧数据有效性（避免无效数据流入下游）"""
        # 1. 数据非空判断
        if color_data is None or depth_data is None:
            return False
        # 2. 数据尺寸合法判断（非空数组，宽高大于0）
        if color_data.shape[0] <= 0 or color_data.shape[1] <= 0:
            return False
        if depth_data.shape[0] <= 0 or depth_data.shape[1] <= 0:
            return False
        # 3. 深度图与彩色图尺寸匹配判断
        if color_data.shape[:2] != depth_data.shape[:2]:
            logger.warning(f"相机 {self.serial_number} 帧数据尺寸不匹配：彩色图{color_data.shape[:2]}，深度图{depth_data.shape[:2]}")
            return False
        # 4. 数据类型合法判断（深度图uint16，彩色图uint8）
        if depth_data.dtype != np.uint16 or color_data.dtype != np.uint8:
            logger.warning(f"相机 {self.serial_number} 帧数据类型不合法：彩色图{color_data.dtype}，深度图{depth_data.dtype}")
            return False
        return True

    def _put_frame_to_queue(self, image_frame: ImageFrame):
        """非阻塞将帧放入队列，队列满时丢弃旧帧（保证线程安全+最新帧）"""
        try:
            # 队列满时，先丢弃最旧帧，再放入新帧
            if self.frame_queue.full():
                old_frame = self.frame_queue.get(block=False)
                logger.debug(f"相机 {self.serial_number} 帧队列已满，丢弃旧帧（时间戳：{old_frame.timestamp}）")
            self.frame_queue.put(image_frame, block=False)
        except queue.Full:
            logger.warning(f"相机 {self.serial_number} 帧队列已满，当前帧丢弃失败")

    def create_capture_thread(self):
        """封装线程创建逻辑（让__enter__更简洁，符合你的要求）"""
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            name=f"CameraCaptureThread-{self.serial_number}",
            daemon=True  # 核心优化：守护线程，避免主线程退出后残留
        )
        self.capture_thread.start()
        logger.info(f"相机 {self.serial_number} 采集线程创建完成")
    
    # 接口
    def get_latest_frame(self, block: bool = True, timeout: int = None) -> ImageFrame or None:
        """
        获取最新图像帧（返回ImageFrame数据结构，线程安全）
        block: True时若无新帧即等待；timeout: None表示无限等待"""
        try:
            return self.frame_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            logger.debug(f"相机 {self.serial_number} 帧队列为空，无最新帧返回")
            return None

    def __enter__(self):
        """上下文管理器进入方法（自动初始化、打开、启动相机）"""
        self.initialize()
        self.open_camera()
        self.start_camera()
        self.create_capture_thread()  # 调用封装的线程创建逻辑，简化__enter__
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器退出方法（自动停止、关闭、释放资源，优化异常打印）"""
        # 1. 释放相机相关资源
        self.stop_camera()
        self.close_camera()
        self.release()

        # 2. 等待采集线程退出（添加超时，避免主线程阻塞）
        if self.capture_thread is not None and self.capture_thread.is_alive():
            logger.info(f"等待相机 {self.serial_number} 采集线程退出...")
            self.capture_thread.join(timeout=5.0)

        # 3. 优化异常打印，保留完整堆栈至日志（核心要求）
        if exc_value is not None:
            error_info = (
                f"相机 {self.serial_number} 取流过程中出现异常\n"
                f"异常类型：{exc_type}\n"
                f"异常信息：{exc_value}\n"
                f"相机状态：初始化={self.is_initialized}，已打开={self.is_opened}，取流中={self.is_streaming}"
            )
            logger.error(error_info, exc_info=True)  # exc_info=True：保留完整异常堆栈
            # 重新抛出异常，让上层感知错误（保留原始异常上下文）
            raise Exception(error_info) from exc_value
    # 工具
    @staticmethod
    def print_camera_list():
        nDeviceNum = c_uint(0)
        nDeviceNum_p = byref(nDeviceNum)
        Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet, nDeviceNum_p)
        print(f"the number of device is: {nDeviceNum.value}")
        if nDeviceNum.value==0:
            print("find no device!")
            raise FileNotFoundError("未找到可用设备，请检查连接和设备占用情况")
        stDeviceList = MV3D_RGBD_DEVICE_INFO_LIST()
        Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_Ethernet, pointer(stDeviceList.DeviceInfo[0]), 10, nDeviceNum_p)
        for i in range(0, nDeviceNum.value):
            strModeName = ""
            for per in stDeviceList.DeviceInfo[i].chModelName:
                strModeName = strModeName + chr(per)
            chSerialNumber = ""
            for per in stDeviceList.DeviceInfo[i].chSerialNumber:
                chSerialNumber = chSerialNumber + chr(per)
            print(f"device: {i}\n\tdevice model name: {strModeName}\n\tdevice SerialName: {chSerialNumber}")

"""
使用案例：
with CameraDriver(serial_number='00DA5939159') as camera:
    while True:
        imageframe = camera.get_latest_frame()
        color_frame, depth_frame = imageframe.color_data, imageframe.depth_data
        if color_frame is not None and depth_frame is not None:
            # 处理图像
            pass
"""
"""
Note:
    1. 目前驱动是根据轮询机制实现的，实际上也可以根据回调注册机制实现。个人认为二者是等价的。
"""