from ctypes import *

import cv2
import open3d as o3d
import numpy as np
from datetime import datetime
import os
import time
import base64

# 内部库
from vision_module.vision_process import *
from vision_module.camera_driver import CameraDriver
from vision_module.data_structures import ImageFrame

from ParcelInfo import *
from utils import constant, path_utils

from collections import deque

paecel_types = {0: 'cardboard_box',
                1: 'cardboard_box_color', 
                2: 'bubble_mailer', 
                3: 'plastic_mailer', 
                4: 'robot'}

class FlowPipeline:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.flow_queue = deque(maxlen=16)   # 用于存储场景图片的队列，16帧约2s
        self.set_state = FlowState.Static  # 初始状态为静态
        self.real_state = FlowState.Static  # 实际状态，初始为静态

        self.previous_frame = None  # 上一帧
        self.current_frame = None   # 当前帧

        self.ratio_history = deque(maxlen=6)   # 用于稳定判断

        # 阈值（后面你再根据现场微调）
        self.T_MOVE = 0.15     # 认为“明显在动”
        self.T_STATIC = 0.05   # 认为“基本静止”
        self.STABLE_N = 4      # 连续N帧稳定

    # 对外接口
    def push_frame(self, frame):
        self.flow_queue.append(frame)
    
    def get_state(self):
        return self.real_state

    def run(self):
        self.current_frame = self.flow_queue.popleft()  # 【TODO: 注意：要保证这里是阻塞等待】
        if self.previous_frame is None:
            self.previous_frame = self.current_frame
            return None
        result = self.flow_process(self.previous_frame, self.current_frame)
        self.previous_frame = self.current_frame
        return result
            
    def flow_process(self, previous_frame, current_frame, roi):
        x,y,w,h = roi
        # 1. 转换为灰度图
        pre_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # 2. 裁剪ROI区域
        pre_gray_roi = pre_gray[y:y+h, x:x+w]
        cur_gray_roi = cur_gray[y:y+h, x:x+w]
        # 3. 计算光流
        flow = cv2.cv2.calcOpticalFlowFarneback(
            pre_gray_roi, cur_gray_roi, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        ratio = np.mean(mag > 1.0)
        self.ratio_history.append(ratio)
        self._update_state(ratio)
        return {
            "ratio": ratio,
            "state": self.real_state,
            "mag_mean": float(mag.mean()),
            "mag_median": float(np.median(mag))
        }
    def _update_state(self, ratio):
        if self.real_state == FlowState.Static:
            if ratio > self.T_MOVE:
                self.logger.info("Flow: detect incoming motion")
                self.real_state = FlowState.Incoming
        # 已检测到上料过程
        elif self.real_state == FlowState.Incoming:
            if self._is_stable():
                self.logger.info("Flow: become stable ready")
                self.real_state = FlowState.StableReady
        elif self.real_state == FlowState.StableReady:
            if ratio > self.T_MOVE:
                self.logger.info("Flow: move again")
                self.real_state = FlowState.Incoming

    def _is_stable(self):
        if len(self.ratio_history) < self.STABLE_N:
            return False
        last = list(self.ratio_history)[-self.STABLE_N:]
        # 1. 都小于静止阈值
        cond1 = all(r < self.T_STATIC for r in last)
        # 2. 波动很小
        cond2 = (max(last) - min(last)) < 0.03
        return cond1 and cond2



class ProcessPipeline:
    def __init__(self, logger, region_manager) -> None:
        self.logger = logger
        self.region_manager = region_manager

        self.frame = None
        self.timestamp = None   # 帧时间戳
        self.color_data = None  # 原始彩色图
        self.depth_data = None  # 原始深度图
        self.detect_result = None  # 包裹检测结果

        self.vision_result = None  # 包裹检测结果

        self.parcel_list = []  # 所有包裹列表
        self.left_parcel_list = []  # 左区包裹列表
        self.right_parcel_list = []  # 右区包裹列表
        self.robot_flag_left = False  # 左区是否有机械臂
        self.robot_flag_right = False  # 右区是否有机械臂

        # 处理过程中的数据载体（类内属性，步骤间共享，无需额外上下文）【TODO: temp】
        # self.color_img = None   # 业务处理用彩色图（避免修改原始数据）
        

        # 性能统计（可选，便于调试）
        self.step_times = {}  # 各步骤耗时
        self.total_time = 0.0  # 总耗时

        # 初始化必要资源（如检测模型，根据你的实际情况调整）
        ## 1. 检测模型
        ## 2. 对比图片
        # resource
        self.obb_model = obb_model  # TODO 这种写法可能并不推荐
        self.compare_img = cv2.imread(path_utils.get_resource_path("compare.png"))

    def put_frame(self, frame):
        self.frame = frame
        self.color_data = frame.color_data
        self.depth_data = frame.depth_data
        self.timestamp = self.frame.timestamp or time.strftime('%Y-%m-%d_%H-%M-%S_%f', time.localtime())

    def run(self, run_cmd):
        """
        运行一次处理流程（从相机取图到包裹检测） 处理一次
        :param run_cmd: 'all', 'left', 'right'
        """
        results = self.obb_model.predict(source=self.color_data,
                        conf=0.7,
                        # max_det=4,
                        iou=0.75,
                        half=True,
                        agnostic_nms=True,
                        device=0)
        self.detect_result = results[0]
        self._sort_region()
        if run_cmd == 'all':
            self.vision_result = self.parcels_parallel_process(self.parcel_list)
        elif run_cmd == 'left':
            self.vision_result = self.parcels_parallel_process(self.left_parcel_list)
        elif run_cmd == 'right':
            self.vision_result = self.parcels_parallel_process(self.right_parcel_list)
        else:
            raise ValueError(f"Invalid run_cmd: {run_cmd}")

    def _sort_region(self):
        for id, box in enumerate(self.detect_result.obb):
            parcel = PackageInfo()
            # 包裹初步封装
            parcel.package_id = f"pkg_{self.timestamp.replace('-', '').replace('_', '')[:14]}_{id:03d}"
            parcel.timestamp = self.timestamp
            parcel.obb = box
            parcel.center_pixel = (box.xywhr[0][0].item(), box.xywhr[0][1].item())
            base_region, sub_region = self.region_manager.get_region_of_point(parcel.center_pixel)
            parcel.base_region_id = base_region.region_id if base_region else "dead_zone"
            parcel.sub_region_id = sub_region.region_id if sub_region else None
            parcel.type = paecel_types[box.cls.item()]
            if base_region and base_region.region_id == 'left_zone':
                self.left_parcel_list.append(parcel)
                self.parcel_list.append(parcel)
            elif base_region and base_region.region_id == 'right_zone':
                self.right_parcel_list.append(parcel)
                self.parcel_list.append(parcel)

    def parcel_process(self, parcel: PackageInfo) -> PackageInfo:
        # TODO:在PackageInfo设置一个字段表示是否正确处理到结尾，默认为False，处理结束为True
        # PackageStatus：涉及到这部分的设计与初始化
        self.logger.debug(f"包裹[{parcel.package_id}] 基础标识信息填充完成：时间戳={parcel.timestamp}，检测索引={parcel.detect_index}")
        parcel.status = PackageStatus.unsolve   # 【提示】：包裹默认状态unsolve(未解算)
        box = parcel.box
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
        if center_3d[2] > self.ddepth + 50:
            parcel.error_msg = f"包裹深度过高，不在分拣台上，相机坐标系depth=={center_3d[2]}"
            return parcel
        # TODO: 下面两条需要重新研判
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
            obb_info = soft_obb_info_v4(roi_pcd_vds, robot_arm = parcel.arm_id)
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
        parcel.status = PackageStatus.solve
        parcel.width = obb_info['short_edge']; parcel.height = obb_info['long_edge']
        parcel.obb_info = obb_info

        # TODO: 修改为与子区域法向量计算的安全墙操作
        cos_angle = filter_normals(obb_info['normal'])
        if cos_angle < 0.707:    # 改为45度
            parcel.status = PackageStatus.ungraspable
            parcel.error_msg = "包裹法向量夹角过大"
            return parcel
        # TODO: 高度参考需要修改 
        if (max(obb_info['long_edge'], self.ddepth - center_3d[2]) > self.parcel_max_size) or (obb_info['short_edge'] < 95):
            parcel.status = PackageStatus.ungraspable
            parcel.error_msg = "包裹尺寸过小"
            return parcel

        # 检查包裹信息的封装[TODO]
        # 【TODO】(暂时可忽略，代码优先级低)将数据结构应用在tranform等方法中
        # 【TODO: 绘图】相应图片状态
        parcel.status = PackageStatus.graspable
        grasp_point_base = transform_point(obb_info['center'], side.extrinsic)
        stay_point_base = transform_point(obb_info['stay3d'], side.extrinsic)
        up_point_base = transform_point(obb_info['up3d'], side.extrinsic)
        grasp_euler_base = transform_orientation(obb_info['R'], side.extrinsic)
        parcel.grasp_point.x,parcel.grasp_point.y,parcel.grasp_point.z = grasp_point_base
        parcel.stay_point.x,parcel.stay_point.y,parcel.stay_point.z = stay_point_base
        parcel.up_point.x,parcel.up_point.y,parcel.up_point.z = up_point_base
        parcel.r.rx,parcel.r.ry,parcel.r.rz = grasp_euler_base
        return parcel

    def parcels_parallel_process(self, parcel_list: List[PackageInfo]) -> List[PackageInfo]:
        """
        多线程并行处理包裹列表，每个包裹独立处理。
        """
        if not parcel_list:
            return []
        max_workers = 5
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

        ######################################################
        color_img, _ = draw_3d_obb_on_image(self.color_img, obb_info['rect3d'], self.intrinsics, color=(0, 255, 0), label="Grasp OBB")
        color_img = draw_graspinfo(color_img, obb_info['center_v'], obb_info['u_axis'], obb_info['v_axis'], self.intrinsics)
        center = project(center_3d, self.intrinsics)
        cv2.circle(color_img, center, 4, (0, 255, 0), -1)
        self.color_img = color_img
        parcel.out_path = self.timestamp
 





class VisionModule:
    def __init__(self, log_callback ,frame_callback):
        # extern callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        # 常量配置
        self.SERIAL_NUMBER = '00DA5939159'
        self.intrinsics = {'fx': 845.8645, 'fy': 849.886841, 'cx': 712.9507, 'cy': 546.7291}
        self.logger = Vision_logger

        # 区域管理器
        self.region_config_path: str = "./region_config.yaml"   # [TODO: 修改为绝对路径]
        self.region_manager = RegionManager(config_path=self.region_config_path)
        # 相机驱动层
        self.camera = CameraDriver(self.SERIAL_NUMBER, frame_queue_size=1, fetch_frame_timeout=5000)  # 持有CameraDriver实例
        self.camera.__enter__() # 启动相机

        # 两个消费端，一个生产端，三个线程
        self.process_task_queue = queue.Queue(maxsize=10)
        self.flow_task_queue = queue.Queue(maxsize=10)
        self.process_thread = Thread(target=self._process_consumer_func, daemon=True)
        self.flow_thread = Thread(target=self._flow_consumer_func, daemon=True)
        self.process_thread.start()
        self.flow_thread.start()
        
        # 处理流程初始化
        self.process_pipeline = ProcessPipeline(self.logger, self.region_manager)
        self.flow_pipeline = FlowPipeline(self.logger)
    
    def _process_consumer_func(self):
        """process_pipeline对应的消费者线程函数（持续监听任务队列）"""
        while True:
            try:
                # 阻塞获取队列中的任务（无任务时挂起，有任务时自动唤醒）
                # task是一个元组：(拷贝后的frame, 执行指令all/left/right)
                frame_copy, run_cmd = self.process_task_queue.get()
                
                # 执行process_pipeline的核心逻辑（对应你原有代码）
                self.process_pipeline.put_frame(frame_copy)
                self.process_pipeline.run(run_cmd)
                
                # 标记任务处理完成（用于队列的task_done()/join()机制，可选但推荐）
                self.process_task_queue.task_done()
                
            except Exception as e:
                # 异常捕获：防止单个任务报错导致整个消费者线程崩溃
                print(f"process_pipeline处理任务失败：{e}")
                continue

    def _flow_consumer_func(self):
        """flow_pipeline对应的消费者线程函数（持续监听任务队列）"""
        while True:
            try:
                # 阻塞获取队列中的任务
                frame_copy, run_cmd = self.flow_task_queue.get()
                
                # 执行flow_pipeline的核心逻辑（对应你原有代码）
                self.flow_pipeline.put_frame(frame_copy)
                self.flow_pipeline.run(run_cmd)
                
                # 标记任务处理完成
                self.flow_task_queue.task_done()
                
            except Exception as e:
                # 异常捕获：保证线程健壮性
                print(f"flow_pipeline处理任务失败：{e}")
                continue

    def main_loop(self):
        while True:
            # 1.取图 2.根据区域状态信息将图片放入对应处理pipeline中与队列中 3.线程并行处理
            ## TODO 线程如何空闲与正确并行释放
            frame = self.camera.get_latest_frame()
            # 获取区域状态信息
            # TODO: 任务队列是不是可以从左右分区出发？
            left_status = self.region_manager.get_region_status("left_zone")
            right_status = self.region_manager.get_region_status("right_zone")

            frame_copy1 = copy.deepcopy(frame)
            frame_copy2 = copy.deepcopy(frame)
            if left_status is RegionStatus.SORT:
                self.process_task_queue.put((frame_copy1, "left"))
            elif left_status is RegionStatus.UP:
                self.flow_task_queue.put((frame_copy2, "left"))
            elif left_status is RegionStatus.REMOVE:
                pass
            
            if right_status is RegionStatus.SORT:
                self.process_task_queue.put((frame_copy1, "right"))
            elif right_status is RegionStatus.UP:
                self.flow_task_queue.put((frame_copy2, "right"))

            # 4.负责处理结果的流程：与决策模块沟通结果
            # 决策模块也需要添加一个任务队列，也在主循环中阻塞处理
            # 传图模块
            # [TODO]
        

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