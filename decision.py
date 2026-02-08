# 包外部库
from robot_controller import PostalDas
from vision import VisionModule
from decision_module.utils import constant, VisionInspectionStrategy
# 包内部库
from decision_module.decision_log import decision_logger
# from decision_module.config import decision_config as C
# from vision_module.vision_process.image_process import compute_depth_diff, mask_left
from vision_module.data_structures import VisionResult, PackageInfo, PackageStatus
from vision_module.region_manager import RegionStatus

# 标准库与第三方库
import time
import math
import copy
import traceback
import threading
from enum import Enum, auto
from queue import Queue, Empty
from collections import Counter

class ZoneIntent(Enum):
    SORT   = auto() # 分拣 (有可抓取包裹)
    UP   = auto() # 上料 (无包裹或包裹不够)
    REMOVE = auto() # 剔除 (只有废料/不可抓取物)

class SystemStrategy(Enum):
    # RUNNING 状态
    L_SORT_R_SORT   = "L-sort_R-sort"   # 策略一：双臂分拣
    L_SORT_R_UP     = "L-sort_R-up"     # 策略二：左抓右补
    L_REMOVE_R_UP   = "L-remove_R-up"   # 策略三：左剔右补 (原策略三逻辑调整)
    L_UP_R_UP       = "L-up_R-up"       # 策略四：双上料
    # 特殊状态
    INITIALIZING    = "Initializing"
    EMERGENCY_STOP  = "Emergency_Stop"


class DecisionModule:

    def __init__(self, vision_module: VisionModule, postal_DAS: PostalDas):
        self.vision_module = vision_module
        self.postal_DAS = postal_DAS
        self.logger = decision_logger # TODO: [可以需要修改]

        self.left_catch_queue = Queue()
        self.right_catch_queue = Queue()
        self.left_sorted_parcels_list = []
        self.right_sorted_parcels_list = []

        # 记录已经被抓取的包裹，上料时清空列表
        self.has_catched_left_list = []
        self.has_catched_right_list = []

        # 状态管理：核心扩展（急停+恢复）
        self.current_state = SystemStrategy.INITIALIZING  # 初始状态
        self._emergency_stop_flag = False # 急停标志（暂停）
        self._emergency_stop_lock = threading.Lock() # 状态锁（保证急停/恢复线程安全）

        self.left_zone_state = ZoneIntent.SORT
        self.right_zone_state = ZoneIntent.SORT

        # 执行器状态
        self.left_up_material_flag = False  # True: 执行器正在运动; False: 执行器停止运动
        self.right_up_material_flag = False # True: 执行器正在运动; False: 执行器停止运动
        self.remove_flag = False            # True: 执行器正在运动; False: 执行器停止运动

    # ---------------------------------------------------------
    # Producer: 视觉模块负责生产视觉结果，通过调用get_latest_result()方法获取
    # ---------------------------------------------------------  
    # self.vision_module.get_latest_result() 会返回最新的视觉结果

    # ---------------------------------------------------------
    # Intermediary & Arbitrator: 决策主逻辑 
    # ---------------------------------------------------------
    def make_decision(self):
        while True:
            if self._emergency_stop_flag:
                time.sleep(0.05)    # 防止死循环空转占用CPU
                continue
            try:
                vision_result = self.vision_module.get_latest_result()
                left_result, right_result = vision_result
                self._vision_result_infer(left_result)
                self._vision_result_infer(right_result)
                
                if self.left_zone_state == ZoneIntent.UP:
                    self.set_right_zone_state(ZoneIntent.UP)

                if self.left_zone_state == ZoneIntent.SORT:
                    self._put_parcels2left_queue(self.left_sorted_parcels_list)
                if self.right_zone_state == ZoneIntent.SORT: 
                    self._put_parcels2right_queue(self.right_sorted_parcels_list)

                # 进行执行器操作
                self._actuator_control()
                # 从左右两区的状态，推理出全局状态
                self.get_scene_state()
                # TODO: 这里接入前端状态显示函数


            except Exception as e:
                tb = traceback.format_exc()
                raise f"make_decision 出现错误: {e}\n{tb}"

    def _vision_result_infer(self, result: VisionResult):
        # cmd = result.cmd      # [NOTE] 注释代码
        counter, sorted_parcels = self._resolve_parcel_list(result.parcel_list)
        if result.region_id == 'left_zone':
            set_zone_state_func = self.set_left_zone_state
            cmd = self.vision_module.get_region_status('left_zone')
            self.left_sorted_parcels_list = sorted_parcels
            catch_queue = self.left_catch_queue
        elif result.region_id == 'right_zone':
            set_zone_state_func = self.set_right_zone_state
            cmd = self.vision_module.get_region_status('right_zone')
            self.right_sorted_parcels_list = sorted_parcels
            catch_queue = self.left_catch_queue
        
        if cmd == RegionStatus.SORT:
            # [sort -> sort] or [sort -> up] or [sort -> remove]
            # 根据parcel.status对parcel_list进行分类
            if counter[ParcelStatus.GRASPABLE] > 0:
                set_zone_state_func(ZoneIntent.SORT)
            else:
                if result.has_robot is True:
                    set_zone_state_func(ZoneIntent.SORT)
                elif result.region_id == 'left_zone' and counter[ParcelStatus.UNGRASPABLE] > 0:
                    set_zone_state_func(ZoneIntent.REMOVE)
                else:
                    set_zone_state_func(ZoneIntent.UP)
            """
            伪代码：
            如果包裹列表中可抓取的包裹数量>0:
                排列优先级，将优先级最高的包裹加入catch_queue
                set_zone_state_func(ZoneIntent.SORT)
            如果列表中可抓取的包裹数量=0:
                如果机械臂遮挡：
                        set_zone_state_func(ZoneIntent.SORT)
                如果存在不可抓取的包裹:
                        set_zone_state_func(ZoneIntent.REMOVE)
                如果不存在不可抓取的包裹:   
                    set_zone_state_func(ZoneIntent.UP)
            
            函数外(make_decision中)设置：如果左边设置为UP，右边直接设置为UP
            """
        elif cmd == RegionStatus.UP:
            # [up -> sort] or [up keeps]
            # [TODO]这里需要修改为视觉部分满足上料需求，暂时修改为如果有包裹可抓取，就设置为SORT
            if best_parcel is not None:
                set_zone_state_func(ZoneIntent.SORT)
            else:
                set_zone_state_func(ZoneIntent.UP)
            # [伪代码]检查上料是否结束
            """
            伪代码：
            检查光流结果
            如果上料完成：
                set_zone_state_func(ZoneIntent.SORT)
            如果上料未完成 -> 保持上料
                set_zone_state_func(ZoneIntent.UP)  # 应当不用设置，依然为上料状态
            """
        elif cmd == RegionStatus.REMOVE:
            # [remove -> up] or [remove keeps]
            if self.remove_flag is False: # 剔除结束
                set_zone_state_func(ZoneIntent.UP)

    def _resolve_parcel_list(self, parcel_list: list[PackageInfo]):
        counter = Counter()
        graspable_parcels = []
        for parcel in parcel_list:
            counter[parcel.status] += 1
            if parcel.status == PackageStatus.GRASPABLE:
                graspable_parcels.append(parcel)
        # 根据parcel的属性对可抓包裹进行打分排序，最终选择最高得分的包裹
        if not graspable_parcels:
            return counter, []
        # 2. 提取特征数据用于归一化
        # 我们需要暂存每个包裹的特征值：(height, size, distance)
        # 假设: 
        #   Height = parcel.grasp_point.z
        #   Size = parcel.obb_info['long_edge'] * parcel.obb_info['short_edge']
        #   Distance = sqrt(x^2 + y^2) (平面欧氏距离) parcel.grasp_point.x and parcel.grasp_point.y
        # [TODO] 需要判定机械臂基座原点坐标为(0, 0, 0)，距离基座由远到近
        features = []
        for parcel in graspable_parcels:
            height = parcel.grasp_point.z
            area = parcel.obb_info['long_edge'] * parcel.obb_info['short_edge']
            dist = math.sqrt(parcel.grasp_point.x**2 + parcel.grasp_point.y**2)
            features.append({'h': height, 's': area, 'd': dist, 'parcel': parcel})
        max_h = max([f['h'] for f in features]); min_h = min([f['h'] for f in features])
        max_s = max([f['s'] for f in features]); min_s = min([f['s'] for f in features])
        max_d = max([f['d'] for f in features]); min_d = min([f['d'] for f in features])

        W_HEIGHT = 0.4; W_SIZE = 0.1; W_DIST = 0.5
        for item in features:
            norm_h = (item['h'] - min_h) / (max_h - min_h) if max_h > min_h else 0
            norm_s = (item['s'] - min_s) / (max_s - min_s) if max_s > min_s else 0
            norm_d = (item['d'] - min_d) / (max_d - min_d) if max_d > min_d else 0
            score = W_HEIGHT * (1 - norm_h) + W_SIZE * norm_s + W_DIST * norm_d
            item['score'] = score
            # [可选] 将分数绑定回 Parcel 对象，方便后续调试查看
        # 3. 选择得分最高的包裹
        features.sort(key=lambda x: x['score'], reverse=True)
        sorted_parcels = [item['parcel'] for item in features]
        return counter, sorted_parcels

    def set_left_zone_state(self, state: ZoneIntent):
        self.logger.info(f"左区状态变更：{self.left_zone_state} → {state}")
        self.left_zone_state = state
        region_status = self._intent2region_status(state)
        self.vision_module.set_region_status('left_zone', region_status)
    def set_right_zone_state(self, state: ZoneIntent):
        self.logger.info(f"右区状态变更：{self.right_zone_state} → {state}")
        self.right_zone_state = state
        region_status = self._intent2region_status(state)
        self.vision_module.set_region_status('right_zone', region_status)
    def get_scene_state(self) -> SystemStrategy:
        # TODO: 这里可能出现第五种状态：L_REMOVE_R_SORT
        if self.left_zone_state == ZoneIntent.SORT and self.right_zone_state == ZoneIntent.SORT:
            self.current_state = SystemStrategy.L_SORT_R_SORT
        elif self.left_zone_state == ZoneIntent.SORT and self.right_zone_state == ZoneIntent.UP:
            self.current_state = SystemStrategy.L_SORT_R_UP
        elif self.left_zone_state == ZoneIntent.REMOVE and self.right_zone_state == ZoneIntent.UP:
            self.current_state = SystemStrategy.L_REMOVE_R_UP
        elif self.left_zone_state == ZoneIntent.UP and self.right_zone_state == ZoneIntent.UP:
            self.current_state = SystemStrategy.L_UP_R_UP
        self.logger.info(f"当前场景状态更新为：{self.current_state}")
        return self.current_state
        
    def _actuator_control(self):
        if self.left_zone_state == ZoneIntent.SORT:
            self.set_actuator_left_up_material(status=False)
        elif self.left_zone_state == ZoneIntent.UP:
            self.set_actuator_left_up_material(status=True)
        elif self.left_zone_state == ZoneIntent.REMOVE:
            self.set_actuator_remove_material()

        if self.right_zone_state == ZoneIntent.SORT:
            self.set_actuator_right_up_material(status=False)
        elif self.right_zone_state == ZoneIntent.UP:
            self.set_actuator_right_up_material(status=True)

    # ---------------------------------------------------------
    # Actuator: 执行器，底层调用
    # ---------------------------------------------------------
    def set_actuator_left_up_material(self, status: bool):
        # True: 上料; False: 停料
        if self.left_up_material_flag == status:
            return
        self.left_up_material_flag = status
        self.has_catched_left_list.clear()
        self.postal_DAS.up_material(id=2, status=status)
        time.sleep(0.3)    # 继电器响应时间 TODO: 移栽机上料响应时间未测试
    def set_actuator_right_up_material(self, status: bool):
        # True: 上料; False: 停料
        if self.right_up_material_flag == status:
            return
        self.right_up_material_flag = status
        self.has_catched_right_list.clear()
        self.postal_DAS.up_material(id=1, status=status)
        time.sleep(0.5)    # 继电器响应时间 TODO: 目前直接写死，后续需要整理到参数文件中
    def set_actuator_remove_material(self):
        if self.remove_flag is True:
            return
        remove_thread = threading.Thread(target=self.thread_remove_material)
        remove_thread.start()
    def thread_remove_material(self):
        try:
            self.postal_DAS.remove_material(status=True)
            self.remove_flag = True
            time.sleep(constant.remove_time)    # 配置参数设定的剔除时间
            self.postal_DAS.remove_material(status=False) # 【TODO】: 具体底层接口需要查看postal_DAS具体实现与设定
            self.remove_flag = False
        except Exception as e:
            tb = traceback.format_exc()
            raise f'剔除包裹出现错误: {e}\n{tb}'

    # ---------------------------------------------------------
    # Customer: 机械臂消费端，提供给机械臂调用的接口
    # ---------------------------------------------------------
    def _put_parcels2left_queue(self, sorted_parcels_list: list):
        if self.left_catch_queue.qsize() >= 2:
            return
        need_num = 2 - self.left_catch_queue.qsize()
        for parcel in sorted_parcels_list:
            if parcel in self.has_catched_left_list:    # TODO: 此处是一个条件，剔除重复包裹
                continue
            self.left_catch_queue.put(parcel)
            need_num -= 1
            if need_num <= 0:
                break
    def _put_parcels2right_queue(self, sorted_parcels_list: list):
        if self.right_catch_queue.qsize() >= 2:
            return
        need_num = 2 - self.right_catch_queue.qsize()
        for parcel in sorted_parcels_list:
            if parcel in self.has_catched_right_list:    # TODO: 此处是一个条件，剔除重复包裹
                continue
            self.right_catch_queue.put(parcel)
            need_num -= 1
            if need_num <= 0:
                break
    # 【TODO】 暂时如此
    def vision_inspection(self, robot_id):
        stime=time.time()
        if robot_id == constant.LEFT_ROBOT_ARM:   # 左臂 1
            self.logger.debug(f"左臂询问包裹，当前队列长度为{self.left_catch_queue.qsize()}")
            parcel = self.left_catch_queue.get()
            self.logger.debug(f"向左臂传递包裹，当前队列长度为{self.left_catch_queue.qsize()}")
            self.has_catched_left_list.append(parcel)
            return stime,time.time(),(VisionInspectionStrategy.NORMAL_OUTPUT, parcel)
        elif robot_id == constant.RIGHT_ROBOT_ARM: # 右臂 2
            self.logger.debug(f"右臂询问包裹，当前队列长度为{self.right_catch_queue.qsize()}")
            parcel = self.right_catch_queue.get()
            self.logger.debug(f"向右臂传递包裹，当前队列长度为{self.right_catch_queue.qsize()}")
            self.has_catched_right_list.append(parcel)
            return stime,time.time(),(VisionInspectionStrategy.NORMAL_OUTPUT, parcel)

    # ---------------------------------------------------------
    # 前端交互
    # ---------------------------------------------------------
    # 1. 控制急停与恢复
    def set_emergency_stop(self):
        with self._emergency_stop_lock:
            if self._emergency_stop_flag:
                return
            self._emergency_stop_flag = True
            self.current_state = SystemStrategy.EMERGENCY_STOP

            self.set_actuator_left_up_material(status=False)
            self.set_actuator_right_up_material(status=False)

            self.has_catched_left_list.clear()
            self.has_catched_right_list.clear()
            self._clear_queue(self.left_catch_queue)
            self._clear_queue(self.right_catch_queue)

            # [TODO] 控制视觉模块相关暂停操作
            self.logger.critical("设置急停状态成功，决策模块完成急停")

    def recover_from_emergency_stop(self):
        with self._emergency_stop_lock:
            if not self._emergency_stop_flag:
                return
            self._emergency_stop_flag = False
            # [TODO] 控制视觉模块相关恢复状态

            self.left_zone_state = ZoneIntent.SORT
            self.right_zone_state = ZoneIntent.SORT
            self.current_state = SystemStrategy.INITIALIZING
            self.logger.critical("恢复正常状态成功，决策模块完成恢复")
    # 2. 向前端传递状态的接口

    # ---------------------------------------------------------
    # 辅助函数
    # 工具函数
    # ---------------------------------------------------------
    @staticmethod
    def _clear_queue(q: Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

    @staticmethod
    def _intent2region_status(intent: ZoneIntent) -> RegionStatus:
        if intent == ZoneIntent.UP:
            return RegionStatus.UP
        elif intent == ZoneIntent.SORT:
            return RegionStatus.SORT
        elif intent == ZoneIntent.REMOVE:
            return RegionStatus.REMOVE
        return RegionStatus.SORT

if __name__ == '__main__':
    vision_module = VisionModule()
    postal_DAS = PostalDAS()
    decision_module = DecisionModule(vision_module, postal_DAS)

    decision_thread = threading.Thread(target=decision_module.make_decision, daemon=True)
    decision_thread.start()

    # 主线程
    while True:
        time.sleep(0.1)

    # # 若单独测试
    # while True:
    #     decision_module.make_decision()
    #     time.sleep(0.05)    # 防止CPU长时间占用

"""
下次直接可以数据驱动，进行强化学习决策
进行数据决策
"""
# class DecisionModule:

    

#         # 等待左右臂完成抓取 # TODO: 确认这个循环是否需要
#         # while (self.postal_DAS.right_robot.packpose_flag or self.postal_DAS.left_robot.packpose_flag):  
#         # TODO: 确认这个循环是否需要
#         while (self.postal_DAS.right_robot.packpose_flag):
#         # while self.postal_DAS.right_robot.in_camera_cover_flag:
        
#         # TODO: 调试代码如何添加
    
#         # 等待机械臂完成抓取移开视野范围
#         # packpose_flag 机械臂是否完成抓取
#         # in_camera_cover_flag 机械臂是否在相机视野内
        

# # 1. 智能上料：连续几帧之间或者几个时间段内，视野内没有变化，则可以进行停止上料 -> 即光流上料
# # 3. 可能拍照时机需要修改为队列变为0，且机械臂移开视野之后的第一时间，这样效率最高

# has_catched_left_list 能否直接调vision_module的数据 clear能不能继续用
# 或者让视觉调这边
########################
# 一、 视觉模块的线程 1. 视觉主线程 2. 处理线程池 3. 相机驱动线程
#   视觉主线程负责调用处理线程池，数据驱动，会因相机驱动线程的数据而阻塞
#   因此视觉这边的急停设置只需要设置相机驱动线程的急停即可（TODO: 设置相机驱动线程的阻塞与恢复）