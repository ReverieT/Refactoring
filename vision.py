import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

import copy
import base64
import time
import queue
from ctypes import *
from threading import Thread
from collections import deque
from typing import List, Tuple, Optional, Dict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime

# 内部库
from vision_module.vision_process import *
from vision_module.camera_driver import CameraDriver
from vision_module.data_structures import ImageFrame, PackageInfo, PackageStatus, VisionResult
from vision_module.region_manager import RegionManager, ValidSceneStatus, RegionStatus


extrinsic = np.array([[0.99996193, -0.0087372, 0.0040242, 0.0207509],
                      [0.00874224, 0.99996229, -0.0040122, -0.0207462],
                      [-0.0040195, 0.0040284, 0.9999626, 0.0207415],
                      [0.0, 0.0, 0.0, 1.0]])
paecel_types = {0: 'cardboard_box',
                1: 'cardboard_box_color', 
                2: 'bubble_mailer', 
                3: 'plastic_mailer', 
                4: 'robot'}

class ProcessPipeline:
    def __init__(self, logger, region_manager, intrinsics) -> None:
        self.logger = logger
        self.region_manager = region_manager
        self.intrinsics = intrinsics

        self.frame = None
        self.timestamp = None   # 帧时间戳
        self.color_data = None  # 原始彩色图
        self.depth_data = None  # 原始深度图
        self.detect_result = None  # 包裹检测结果

        self.vision_result = None  # 包裹检测结果

        self.left_parcel_list = []  # 左区包裹列表
        self.right_parcel_list = []  # 右区包裹列表
        self.robot_flag_left = False  # 左区是否有机械臂
        self.robot_flag_right = False  # 右区是否有机械臂

        # 处理过程中的数据载体（类内属性，步骤间共享，无需额外上下文）【TODO: temp】
        self.color_img = None   # 每帧处理彩色图，最终上传前端

        # 性能统计（可选，便于调试）
        self.step_times = {}  # 各步骤耗时
        self.total_time = 0.0  # 总耗时

        # 初始化必要资源 1. 模型【TODO】
        ## 2. 对比图片
        self.obb_model = YOLO(OBB_MODEL_PATH)
        warm_up(self.obb_model)

        # self.compare_img = cv2.imread(path_utils.get_resource_path("compare.png"))

    def put_frame(self, frame):
        self.frame = frame
        self.color_data = frame.color_data
        self.depth_data = frame.depth_data
        self.color_img = self.color_data    # TODO: 先不拷贝，之后试试浅拷贝
        self.timestamp = self.frame.timestamp or time.strftime('%Y-%m-%d_%H-%M-%S_%f', time.localtime())

    def run(self):
        """
        运行一次处理流程（从相机取图到包裹检测） 处理一次
        :param run_cmd: 'all', 'left', 'right'
        """
        # 清空变量
        t0 = time.time()
        self.robot_flag_left = False  # 左区是否有机械臂
        self.robot_flag_right = False  # 右区是否有机械臂
        self.left_parcel_list.clear()
        self.right_parcel_list.clear()
        results = self.obb_model.predict(source=self.color_data,    # 【TODO】 参数需要测试，可以使用x-anylabeling测试
                        conf=0.7,
                        iou=0.5,
                        half=True,
                        agnostic_nms=True,
                        device=0)
        self.detect_result = results[0]
        self._sort_region()
        left_list = self.parcels_parallel_process(self.left_parcel_list)
        right_list = self.parcels_parallel_process(self.right_parcel_list)
        cost = (time.time() - t0) * 1000
        self.logger.info(f"[性能] 单帧处理耗时: {cost:.1f}ms | 左区:{len(left_list)} 右区:{len(right_list)} 左区机械臂:{self.robot_flag_left} 右区机械臂:{self.robot_flag_right}")
        return left_list, right_list, self.robot_flag_left, self.robot_flag_right, self.color_img

    def _sort_region(self):
        for id, box in enumerate(self.detect_result.obb):
            parcel = PackageInfo(timestamp=self.timestamp, obb=box)
            # 包裹初步封装
            parcel.package_id = f"pkg_{self.timestamp.replace('-', '').replace('_', '')[:14]}_{id:03d}"
            parcel.center_pixel = (box.xywhr[0][0].item(), box.xywhr[0][1].item())
            base_region, sub_region = self.region_manager.get_region_of_point(parcel.center_pixel)
            parcel.base_region_id = base_region.region_id if base_region else "dead_zone"
            parcel.sub_region_id = sub_region.sub_region_id if sub_region else None
            parcel.type = paecel_types[box.cls.item()]
            if base_region and base_region.region_id == 'left_zone':
                # TODO: 后续修改为不直接使用数字，将这些常数放在vision_utils中
                parcel.arm_id = 1
                self.left_parcel_list.append(parcel)
            elif base_region and base_region.region_id == 'right_zone':
                parcel.arm_id = 2
                self.right_parcel_list.append(parcel)

    def parcel_process(self, parcel: PackageInfo) -> PackageInfo:
        # PackageStatus：表示包裹状态，从是否成功处理到是否可以抓取，状态一直被跟踪
        self.logger.debug(f"包裹[{parcel.package_id}] 基础标识信息填充完成：时间戳={parcel.timestamp}")
        parcel.status = PackageStatus.UNSOLVE   # UNSOLVE：未解算
        box = parcel.obb
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.color_data.shape[1], x2)
        y2 = min(self.color_data.shape[0], y2)
        roi_depth = self.depth_data[y1:y2, x1:x2]
        roi_color = self.color_data[y1:y2, x1:x2]

        mask = obb_mask(roi_depth, box)
        roi_depth = cv2.bitwise_and(roi_depth, roi_depth, mask=mask)
        roi_pcd, str = depth2pointcloud(x1, y1, roi_depth, self.intrinsics)
        if roi_pcd is None:
            parcel.error_msg = "无法获取点云，可能是机械臂占据"+str
            return parcel
        roi_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        # TODO: 需要判断各种异常情况出现的频率，这里也许需要修改一下
        center_3d = get_depth_center3d(x1, y1, roi_depth, self.intrinsics)
        if center_3d is None:
            parcel.error_msg = "无法获取包裹中心点对应的3D坐标"
            return parcel
        # TODO: 这里需要修改为使用参考图像来进行判断
        # if center_3d[2] > self.ddepth + 50:
        #     parcel.error_msg = f"包裹深度过高，不在分拣台上，相机坐标系depth=={center_3d[2]}"
        #     return parcel
        # TODO: 下面两条需要重新研判
        self.logger.debug(f"[深度] 包裹 {parcel.package_id} 相机Z距离: {center_3d[2]:.1f}mm")
        roi_pcd_vds = roi_pcd.voxel_down_sample(voxel_size=10)  # 0.01 *1000
        roi_pcd_vds = normal_cluster(roi_pcd_vds)
        if roi_pcd_vds is None:
            parcel.error_msg = "normal_cluster 失败，筛出后点云为零"
            return parcel
        if parcel.type == 'cardboard_box' or parcel.type == 'cardboard_box_color': # hard_parcel
            plane_pcd, plane_model = ransac_plane_cluster(roi_pcd_vds, distance_threshold=0.005*1000)  # 0.005m
            obb_info = plane2obb(plane_pcd, plane_model, robot_arm = parcel.arm_id)
            if obb_info is None:
                parcel.error_msg = "hard_parcel识别错误"
                return parcel
        elif parcel.type == 'bubble_mailer' or parcel.type == 'plastic_mailer': # soft_parcel
            obb_info = soft_obb_info(roi_pcd_vds, robot_arm = parcel.arm_id)
            if obb_info is None:
                parcel.error_msg = "soft_parcel识别错误"
                return parcel
        elif parcel.type == 'robot':
            # 【TODO】：目前为最简操作，后续可以添加贡多鲁棒操作，比如判断机械臂中心位置等
            if parcel.arm_id == 'left':
                self.robot_flag_left = True
            elif parcel.arm_id == 'right':
                self.robot_flag_right = True
            parcel.error_msg = "非包裹为机械臂"
            return None
        # 至此，包裹解算完成，置为解算状态
        parcel.status = PackageStatus.SOLVE
        parcel.width = obb_info['short_edge']; parcel.height = obb_info['long_edge']
        parcel.obb_info = obb_info

        # TODO: 修改为与子区域法向量计算的安全墙操作
        cos_angle = filter_normals(obb_info['normal'])
        if cos_angle < 0.8:    # 改为45度
            parcel.status = PackageStatus.UNGRASPABLE
            parcel.error_msg = "包裹法向量夹角过大"
            return parcel
        # TODO: 高度参考需要修改 
        # if (max(obb_info['long_edge'], self.ddepth - center_3d[2]) > self.parcel_max_size) or (obb_info['short_edge'] < 95):
        # if max(obb_info['long_edge'], obb_info['short_edge']) > self.parcel_max_size:
        if max(obb_info['long_edge'], obb_info['short_edge']) < 100:
            parcel.status = PackageStatus.UNGRASPABLE
            parcel.error_msg = "包裹尺寸过小"
            return parcel

        # 检查包裹信息的封装[TODO]
        # 【TODO】(暂时可忽略，代码优先级低)将数据结构应用在tranform等方法中
        # 【TODO: 绘图】检查绘图是否需要重构
        color_img, _ = draw_3d_obb_on_image(self.color_img, obb_info['rect3d'], self.intrinsics, color=(0, 255, 0), label="Grasp OBB")
        color_img = draw_graspinfo(color_img, obb_info['center_v'], obb_info['u_axis'], obb_info['v_axis'], self.intrinsics)
        self.color_img = color_img
        parcel.status = PackageStatus.GRASPABLE
        grasp_point_base = transform_point(obb_info['center'], extrinsic)
        stay_point_base = transform_point(obb_info['stay3d'], extrinsic)
        up_point_base = transform_point(obb_info['up3d'], extrinsic)
        grasp_euler_base = transform_orientation(obb_info['R'], extrinsic)
        parcel.grasp_point.x,parcel.grasp_point.y,parcel.grasp_point.z = grasp_point_base
        parcel.stay_point.x,parcel.stay_point.y,parcel.stay_point.z = stay_point_base
        parcel.up_point.x,parcel.up_point.y,parcel.up_point.z = up_point_base
        parcel.euler_angle.rx,parcel.euler_angle.ry,parcel.euler_angle.rz = grasp_euler_base
        self.logger.info(f"[结果] 包裹 {parcel.package_id}: "
                 f"抓取点 World(x={parcel.grasp_point.x:.1f}, y={parcel.grasp_point.y:.1f}, z={parcel.grasp_point.z:.1f}) | "
                 f"角度 Rz={parcel.euler_angle.rz:.2f}")
        return parcel

    def parcels_parallel_process(self, parcel_list: List[PackageInfo]) -> List[PackageInfo]:
        """
        多线程并行处理包裹列表，每个包裹独立处理。
        """
        if not parcel_list:
            return []
        max_workers = 10
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_parcel = {executor.submit(self.parcel_process, parcel): parcel for parcel in parcel_list}
            for future in as_completed(future_to_parcel):
                parcel = future_to_parcel[future]   # 【INFO】:这一条是为了日志记录
                try:
                    result_parcel = future.result()
                    if result_parcel is not None:   # 【INFO】:这里可以添加排序、剔除等等策略
                        results.append(result_parcel)
                    else:
                        self.logger.error(f"包裹{parcel.package_id}处理失败，返回为None: {parcel.error_msg}")
                except Exception as e:
                    self.logger.error(f"包裹{parcel.package_id}处理期间异常: {parcel.error_msg}, 异常信息: {e}")    
        return results


 





class VisionModule:
    def __init__(self, log_callback ,frame_callback):
        # extern callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        # 常量配置
        self.SERIAL_NUMBER = '00DA5939159'
        self.jpeg_qulity = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        self.logger = Vision_logger

        # 区域管理器
        self.region_config_path: str = "D:/PostalDAS/resources/vision/region_config.yaml"   # [TODO: 修改为绝对路径]
        self.region_manager = RegionManager(config_path=self.region_config_path)
        # 相机驱动层
        self.camera = CameraDriver(self.SERIAL_NUMBER, frame_queue_size=1, fetch_frame_timeout=5000)  # 持有CameraDriver实例
        self.camera.__enter__() # 启动相机
        self.intrinsics = self.camera.get_intrinsics()
        self.intrinsics = {'fx': 845.8645, 'fy': 849.886841, 'cx': 712.9507, 'cy': 546.7291}    # [NOTE]: 调试语句

        # 视觉模块结果队列设置
        self.result_queue = queue.Queue(maxsize=1)  # maxsize=1 时刻保持最新结果

        # 处理流程初始化
        self.process_pipeline = ProcessPipeline(self.logger, self.region_manager, self.intrinsics)
    
    def main_loop(self):
        while True:
            # 1.取图 2.根据区域状态信息将图片放入对应处理pipeline中与队列中 3.线程并行处理
            ## TODO 线程如何空闲与正确并行释放
            frame = self.camera.get_latest_frame()
            left_status = self.region_manager.get_region_status('left_zone')
            right_status = self.region_manager.get_region_status('right_zone')
            show_color_image(frame.color_data)
            self.process_pipeline.put_frame(frame)
            left_list, right_list, robot_flag_left, robot_flag_right, color_img = self.process_pipeline.run()
            # 封装为VisionResult格式，与决策模块统一规范
            # [TODO]封装cmd与has_robot
            left_result = VisionResult(region_id='left_zone', parcel_list=left_list, has_robot=robot_flag_left, cmd=left_status)
            right_result = VisionResult(region_id='right_zone', parcel_list=right_list, has_robot=robot_flag_right, cmd=right_status)
            self._put_result_to_queue((left_result, right_result))
            # =========================================================
            # 前端传图
            # =========================================================
            success, buffer = cv2.imencode('.jpg', color_img, self.jpeg_qulity)
            if success:
                b64_data = base64.b64encode(buffer)     # .encode('utf-8')
                full_data_uri = f"data:image/jpeg;base64,{b64_data}"
                self.frame_callback(full_data_uri)
            else:
                self.logger.warning("前端传图跳过一帧：JPEG编码失败")
            # =========================================================
            # 上报视觉状态  【TODO】：依然需要对接调整，或者把视觉状态反馈放在决策模块，每一次决策之后
            # =========================================================
            status_dict = {
                'log_level': 'info',
                'event_type': 'status',
                'log_source': 'vision',
                'log_status': f'Left:{left_status}, Right:{right_status}',     #【TODO】校验是否正确
            }
            self.log_callback(status_dict)
            # =========================================================
            # 调试部分
            # =========================================================
            # print_debug_report(self.logger, left_list, right_list)
            # show_color_image(color_img)
            # =========================================================


            # 等待本次处理完成，回馈结果后，再取下一帧
            
            # 4.负责处理结果的流程：与决策模块沟通结果
            # 决策模块也需要添加一个任务队列，也在主循环中阻塞处理
            
    def _put_result_to_queue(self, result): #这样简单处理一下就不阻塞了
        if self.result_queue.full():
            self.result_queue.get(block=False)  # 弹出最早的结果，腾出空间
        self.result_queue.put(result)

    # 给决策模块的接口
    def get_lastest_result(self):
        return self.result_queue.get(block=True, timeout=None)  # 阻塞等待最新结果，无超时时间设置（一直等待）






"""
视觉模块与决策模块的职能解耦合
视觉模块最终输出一个视觉场景结果的反馈，这个反馈可以是两个，左区和右区分别是一个反馈结果
这个反馈结果需要包含当前区域场景的全部场景信息，方便决策模块进行决策
即包括决策模块进行决策的全部信息
举例：
当前区域（左区）场景中，一个存在多少个包裹，每个包裹的基本信息是可以拥有的，且了解到不同类别的包裹的基本信息
可抓取的包裹有哪些；
当前场景有没有机械臂的遮挡；（甚至可以检测到机械臂就可以提前丢弃帧）

然后把这个反馈信息发送到决策模块后，决策模块即进行决策
决策模块需要决策出当前场景是否适合抓取，即场景状态的判定：需要抓取/上料/剔除
如果需要抓取：
再从视觉反馈中对可抓包裹进行排序与判定，给每个包裹打分，选择得分最高的包裹进行抓取，次高的包裹可以作为候选
可能需要记录已抓取包裹的位置，直到下一次上料之后清空，避免空抓包裹
还需要考虑到场景情况，避免抓取与其他包裹或者其他部分的碰撞
如果需要上料：
向视觉模块反馈状态，则视觉模块不按照原定计划进行处理（重要重要重要，这里可能会涉及到节拍的设置），进行光流法判断上料
如果需要剔除：
直接剔除，也可以使用光流法判断剔除情况是否可观，进行上报
"""


"""
VisionModule与DecisionModule的信息传递与交互
每次传递一个


VisionModule
每次循环前先查看当前两个区域状态如何，然后根据区域状态选择是否进行处理循环以及如何处理循环
然后进行处理，处理结束后（且确定决策模块已经根据处理结果进行了决策）再进行下一次循环取最新的图，每次都固定本次取图的结果直到循环结束
通过特定的数据结果完成与决策模块的信息传递
"""

"""
测试部分
"""
def print_debug_report(logger, left_parcels: list, right_parcels: list):
    """
    打印本帧视觉处理的详细调试报告
    """
    # ------------------ 1. 统计数据 ------------------
    l_total = len(left_parcels)
    l_ok = sum(1 for p in left_parcels if p.status == PackageStatus.GRASPABLE)
    
    r_total = len(right_parcels)
    r_ok = sum(1 for p in right_parcels if p.status == PackageStatus.GRASPABLE)

    logger.info("=" * 60)
    logger.info(f"【视觉帧报表】总计检测: {l_total + r_total} | 可抓取: {l_ok + r_ok}")
    logger.info("-" * 60)

    # ------------------ 2. 左区详情 ------------------
    logger.info(f"🏛️ [左区 Left] (总数:{l_total}, 可抓:{l_ok})")
    if l_total == 0:
        logger.info("   (空)")
    else:
        for p in left_parcels:
            _log_single_parcel(logger, p)

    logger.info("-" * 30)

    # ------------------ 3. 右区详情 ------------------
    logger.info(f"🏛️ [右区 Right] (总数:{r_total}, 可抓:{r_ok})")
    if r_total == 0:
        logger.info("   (空)")
    else:
        for p in right_parcels:
            _log_single_parcel(logger, p)
            
    logger.info("=" * 60)

def _log_single_parcel(logger, p):
    """辅助函数：打印单个包裹信息"""
    pid = p.package_id.split('_')[-1] # 只显示最后序号，简洁一点，如 '001'
    ptype = p.type
    
    if p.status == PackageStatus.GRASPABLE:
        # 【可抓取】：打印 坐标 (x,y,z) + 角度 (rz)
        # 假设坐标单位是 mm
        pos_str = f"x={p.grasp_point.x:6.1f}, y={p.grasp_point.y:6.1f}, z={p.grasp_point.z:6.1f}"
        angle_str = f"rz={p.euler_angle.rz:6.2f}"
        logger.info(f"   ✅ [OK] ID:{pid} | {ptype:<15} | {pos_str} | {angle_str}")
    else:
        # 【不可抓】：打印 错误原因
        logger.warning(f"   ❌ [NG] ID:{pid} | {ptype:<15} | 原因: {p.error_msg}")

if __name__ == "__main__":
    def frame_callback(frame_base64):
        pass

    def log_callback(info):
        pass

    cv_module = VisionModule(log_callback, frame_callback)
    cv_module.main_loop()
    # # 对于主线程
    # cv_thread = threading.Thread(target=cv_module.main_loop)
    # cv_thread.start()

"""
TODOList:
1. 安全墙修改
2. 删除区域管理器过于复杂的操作，保留必要子区域即可，安全墙可以替代大部分防撞子区域
"""