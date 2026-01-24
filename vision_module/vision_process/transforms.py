import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ['T_base_left_cam', 'T_base_right_cam',
           'transform_point', 'transform_orientation',
           'security_wall' , ]



# 外参矩阵
r00=0.70954548185103394  
r01=-0.70458504844781489 
r02=0.010252740539697287 
r10=-0.70440624321192413 
r11=-0.70960540970529506 
r12=-0.016492635964088662
r20=0.018895864861063563 
r21=0.0044801808859382215
r22=-0.99981141935386086 
t0=706.79987005314251    
t1=-890.77854779051279   
t2=910.95843390205141
T_base_left_cam = np.array([[r00, r01, r02, t0], [r10, r11, r12, t1], [r20, r21, r22, t2], [0.0, 0.0, 0.0, 1.0]])

r00=0.78597166104513594
r01=0.61812121438485312
r02=0.0132178803650395
r10=0.61823741120276421
r11=-0.78595279243098148
r12=-0.0077917558513382543
r20=0.0055723803939019476
r21=0.01429588742740435
r22=-0.99988228116084177
t0=0.6308154497020233
t1=-992.59705203047679
t2=915.20229409750425
T_base_right_cam = np.array([[r00, r01, r02, t0], [r10, r11, r12, t1], [r20, r21, r22, t2], [0.0, 0.0, 0.0, 1.0]])

def transform_point(point_cam, T_base_cam):
    point_cam_h = np.append(point_cam, 1.0)  # 齐次坐标 [x, y, z, 1]
    point_base_h = T_base_cam @ point_cam_h
    point_base = point_base_h[:3]
    return point_base

def transform_orientation(R_grasp_cam, T_base_cam):
    R_base_cam = T_base_cam[:3, :3]
    R_grasp_base = R_base_cam @ R_grasp_cam
    grasp_euler_base = R.from_matrix(R_grasp_base).as_euler('xyz', degrees=True)
    return grasp_euler_base
# a b c
# 左： [-472.5519999999999,470.9796,663886.9063514799]
# 右:  [413.2786000000001,325.8082,295872.86983278004]
left_area_wall = [
    (-472.5519999999999, 470.9796, 663886.9063514799),
]
right_area_wall = [
    (413.2786000000001, 325.8082, 295872.86983278004),
]
def security_wall(point, arm_id):
    """
    安全墙函数，在机械臂的工作范围之外，需要筛出包裹
    坐标系：机械臂坐标系
    input: 包裹坐标 x, y
    output: 安全墙判断结果 True or False
    安全墙定义： plane = [(a, b, c, d)]  a*x+b*y+c*z+d<=0
    """
    if arm_id == 1:
        planes = left_area_wall
    elif arm_id == 2:
        planes = right_area_wall
    point = np.asarray(point)
    planes = np.asarray(planes)
    # 计算所有平面的点积
    values = np.dot(planes[:, :2], point) + planes[:, 2]
    # 检查是否所有值 >= 0 (考虑浮点误差)
    return np.all(values >= 1e-6)


# todo: 设置一个机械臂死区位置的包裹筛除函数，在边缘位置，机械臂的头部点位超出机械臂的工作范围，需要筛出包裹
# todo: 利用帧差法判断是否存在没有检测出来的包裹
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
# 相机的内参矩阵
K_cam = np.array([
    [981.6280517578125, 0, 706.5437622070312],
    [0, 980.9905395507812, 575.2250366210938],
    [0, 0, 1]
])
K_cam_intrinsics = {'fx': 981.6280517578125, 'fy': 980.9905395507812, 'cx': 706.5437622070312, 'cy': 575.2250366210938}


def transform_cam_to_base(grasp_point_cam, stay_point_cam, R_grasp_cam, T_base_cam):
    """
    将抓取点和姿态从相机坐标系转换到机械臂基坐标系

    参数:
        grasp_point_cam: (3,) np.array 相机坐标系下的抓取点 [x, y, z]
        R_grasp_cam: (3, 3) np.array 相机坐标系下的旋转矩阵
        T_base_cam: (4, 4) np.array 相机→机械臂的外参变换矩阵
    返回:
        grasp_point_base: (3,) np.array 在机械臂坐标系下的抓取点
        grasp_euler_base: (3,) np.array 欧拉角 [roll, pitch, yaw]（XYZ顺序，角度）
    """
    # grasp 点转换
    point_cam_h = np.append(grasp_point_cam, 1.0)  # 齐次坐标 [x, y, z, 1]
    point_base_h = T_base_cam @ point_cam_h
    grasp_point_base = point_base_h[:3]
    # stay 点转换
    point_cam_h = np.append(stay_point_cam, 1.0)  # 齐次坐标 [x, y, z, 1]
    point_base_h = T_base_cam @ point_cam_h
    stay_point_base = point_base_h[:3]
    # 姿态转换
    R_base_cam = T_base_cam[:3, :3]
    R_grasp_base = R_base_cam @ R_grasp_cam
    grasp_euler_base = R.from_matrix(R_grasp_base).as_euler('xyz',degrees=True) 
    return grasp_point_base, stay_point_base, grasp_euler_base
# def parcelinfo(parcel, grasp_point_3d, stay_point_3d, grasp_euler):
#     parcel.x, pacel.y, pacel.z = grasp_point_3d
#     pacel.r = euler_angle()
#     pacel.r.rx, pacel.r.ry, pacel.r.rz = grasp_euler
#     pacel.order = 0 # TODO: 需要根据包裹顺序进行赋值
#     pacel.arm_id = 0 # TODO: 需要根据抓取手臂进行赋值
#     return pacel