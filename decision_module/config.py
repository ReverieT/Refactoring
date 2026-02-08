from easydict import EasyDict as edict

decision_config = edict()
C = decision_config

# todo: 决策模块部分参数配置
C.average_time = 2      # 平均一次抓取放料的时间

# 策略二：左区有料，右区无料
C.State_Two = edict()
C.State_Two.belt_delay = 1   # 传送带停止延时

# 策略三：左区无料，右区有料
C.State_Three = edict()
C.State_Three.remove_num = 2        # 左区剔除阈值
C.State_Three.remove_time = 2.6   # 剔除时间
# C.State_Three.remove_time = 2.355   # 剔除时间
C.State_Three.right_catch_num = 1   # 剔除时右区可抓包裹数量阈值


# 策略四：左右区均没有包裹
C.State_Four = edict()
C.State_Four.boxes_num = 6      # 全局包裹数量阈值
C.State_Four.left_num = 2       # 上料左区需要包裹数量
C.State_Four.right_num = 2      # 上料右区需要包裹数量
C.State_Four.out_time = 5       # 上料超时时间
C.State_Four.belt_delay = 1   # 传送带停止延时

        
        