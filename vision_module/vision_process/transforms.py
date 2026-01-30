import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ['T_base_left_cam', 'T_base_right_cam',
           'transform_point', 'transform_orientation',
           'security_wall' , ]

# 外参矩阵
r00=0.70315316361868885  
r01=-0.71072787458662678 
r02=-0.021012300649990358
r10=-0.70923778678166849 
r11=-0.7031679576561     
r12=0.050364522501357062 
r20=-0.050570646565669508
r21=-0.020511255722783686
r22=-0.99850983875703736 
t0=847.33347374733535
t1=-1029.7222755326857
t2=1042.7372012987623
T_base_left_cam = np.array([[r00, r01, r02, t0], [r10, r11, r12, t1], [r20, r21, r22, t2], [0.0, 0.0, 0.0, 1.0]])

r00=0.78526172747188627
r01=0.61606901689685423
r02=-0.061830298297941226
r10=0.61563968002080682
r11=-0.78752872841208954
r12=-0.028040797233954629
r20=-0.065968202580844293
r21=-0.016045820184111471
r22=-0.99769270213972794
t0=157.80806871682202
t1=-876.41372348471793
t2=1044.4007608555683
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