##########################################################
# File: point_cloud_process.py
##########################################################
import open3d as o3d
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils import path_utils
from .vision_log import Vision_logger

__all__ = [
    # base
    'depth2pointcloud',     # 将深度图转换为点云
    'save_point_cloud',     # 保存点云
    # tools
    'project',              # 将3D点投影到2D图像上
    'get_depth_center3d',   # 获取深度图中心点对应的3D点
    # process
    ## preprocess
    'ransac_plane_cluster', # 平面分割
    'plane2obb',            # 从平面点云获取最小外接矩形
    'soft_obb_info',        # 软包obb算法
    'soft_obb_info_v1', 'soft_obb_info_v2', 'soft_obb_info_v3', # 软包策略
    # 'ransac_plane_cluster_test',
    'normal_cluster',   # 软包法向量滤除点云
    'normal_and_cluster',   # 法向量聚类
    ## postprocess
    'filter_normals',       # 法向量滤除
    # visualize
    'draw_3d_obb_on_image',
    'draw_graspinfo',
    # test
    'validate_grasp_support',
    'adjust_soft_obb'
]

# base
def depth2pointcloud(x1, y1, depth, intrinsics):
    # x1, y1: ROI基准坐标
    height, width = depth.shape
    threshold = 356     # 阈值：机械臂头顶高度
    # 2pointcloud
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u + x1 - intrinsics['cx']) * z / intrinsics['fx']
    y = (v + y1 - intrinsics['cy']) * z / intrinsics['fy']
    points = np.dstack((x, y, z)).reshape(-1, 3)
    z = points[:, 2]
    points = points[(z > 500) & (z < 2000)]
    z = points[:, 2]
    # 防止机械臂遮挡，超过阈值的点过高则判断为遮挡，不会计算位姿，放置位姿计算错误
    num_points_below_threshold = np.sum(z < 1060 - threshold)
    ratio = num_points_below_threshold / len(z)
    if ratio >= 0.1:
        str = f"z高度超出{threshold}mm的点占比达到{ratio * 100}%"
        return None, str

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    str = f"z高度超出{threshold}mm的点占比达到{ratio * 100}%"
    return pcd, str
def save_point_cloud(pcd, save_dir="pointcloud_outputs", prefix="pcd", fmt="ply"):
    """
    保存点云文件
    
    参数:
        pcd (o3d.geometry.PointCloud): 要保存的点云对象
        save_dir (str): 保存文件夹
        prefix (str): 文件名前缀
        fmt (str): 文件格式 (ply, pcd, xyz, pts)
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 拼接文件路径
    filename = f"{prefix}_{timestamp}.{fmt}"
    filepath = os.path.join(save_dir, filename)

    # 保存点云
    success = o3d.io.write_point_cloud(filepath, pcd)

    if success:
        print(f"点云已保存: {filepath}")
    else:
        print(f"点云保存失败: {filepath}")

    return filepath

# tools
# 将3D点反投影到相机2D
def project(point3D, intrinsics):
    x, y, z = point3D
    if z <= 0: z = 1e-6  # 避免除以0
    u = (intrinsics['fx'] * x) / z + intrinsics['cx']
    v = (intrinsics['fy'] * y) / z + intrinsics['cy']
    return int(u), int(v)

def get_depth_center3d(x1, y1, roi_depth, intrinsics, patch_size=2):
    height, width = roi_depth.shape
    center_v = height // 2
    center_u = width // 2
    xx1 = max(0, center_u - patch_size)
    xx2 = min(width, center_u + patch_size)
    yy1 = max(0, center_v - patch_size)
    yy2 = min(height, center_v + patch_size)
    depth_values = []
    for v in range(yy1, yy2):
        for u in range(xx1, xx2):
            z = roi_depth[v, u]
            if(z > 500) & (z < 2000):
                depth_values.append(z)
    if len(depth_values) == 0:
        print("深度图无效值导致无法计算中心点对应的3D坐标")
        return None
    z0 = sum(depth_values) / len(depth_values)
    x0 = (center_u + x1 - intrinsics['cx']) * z0 / intrinsics['fx']
    y0 = (center_v + y1 - intrinsics['cy']) * z0 / intrinsics['fy']
    center_3d = np.array([x0, y0, z0])  # 图像中心对应的3D坐标（未过滤）
    return center_3d

# visualize
# 在图像上绘制抓取面的3D OBB边框(长方形边框)
def draw_3d_obb_on_image(image, obb_3d_points, intrinsics, color=(0, 255, 255), label="OBB"):
    """
    在图像上绘制投影后的3D OBB边框
    - obb_3d_points: (4, 3) 的 ndarray，每一行是一个3D点 (X, Y, Z)
    """
    projected_pts = [project(p, intrinsics) for p in obb_3d_points]
    # 绘制边框（OBB 4条边）
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0)  # 顶面
    ]
    for i, j in edges:
        pt1, pt2 = projected_pts[i], projected_pts[j]
        cv2.line(image, pt1, pt2, color, 2)
    center_pt = np.mean(obb_3d_points, axis=0)
    center_2d = project(center_pt, intrinsics)
    cv2.putText(image, label, center_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image, projected_pts

# 在图片上标注包裹抓取点和包裹面姿态
def draw_graspinfo(color_img, center_3d, u_axis_3d, v_axis_3d, intrinsics):
    # 这里的中心点即包裹抓取点
    center = project(center_3d, intrinsics)
    length_long = 40
    length_short = 20
    
    u = project(u_axis_3d, intrinsics)
    # cv2.line(color_img, center, u, (0, 0, 255), 2)
    u = (u[0] - center[0], u[1] - center[1])    # 获得坐标方向
    u = np.array(u, dtype=float)
    u = u / np.linalg.norm(u)
    end_point = (
        int(center[0] + u[0] * length_long),
        int(center[1] + u[1] * length_long)
    )
    cv2.line(color_img, center, end_point, (0, 0, 255), 2)

    v = project(v_axis_3d, intrinsics)
    # cv2.line(color_img, center, v, (0, 0, 255), 2)
    v = (v[0] - center[0], v[1] - center[1])    # 获得坐标方向
    v = np.array(v, dtype=float)
    v = v / np.linalg.norm(v)
    end_point = (
        int(center[0] + v[0] * length_short),
        int(center[1] + v[1] * length_short)
    )
    cv2.line(color_img, center, end_point, (0, 0, 255), 2)

    cv2.circle(color_img, center, 10, (0, 255, 0), -1)
    return color_img
# process
## preprocess
#----------------------------#
# 使用RANSAC算法进行平面分割
def ransac_plane_cluster_test(pcd, center_3d, distance_threshold=0.01*1000, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    def point_to_plane_distance(plane_model, center_3d):
        a, b, c, d = plane_model
        x0, y0, z0 = center_3d
        numerator = abs(a * x0 + b * y0 + c * z0 + d)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        distance = numerator / denominator
        return distance
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True) 
    distance = point_to_plane_distance(plane_model, center_3d)
    inlier_cloud, _ = inlier_cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=2)
    if distance > distance_threshold * 2:  #距离过大再次分割，距离平面超过5cm
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
        inlier_cloud = outlier_cloud.select_by_index(inliers)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
        distance = point_to_plane_distance(plane_model, center_3d)
        if distance > distance_threshold * 2:
            print(f"distance(2)={distance}")
            print("分割错误")
            return None, None
    return inlier_cloud, plane_model
#----------------------------#
def ransac_plane_cluster(pcd, distance_threshold=0.01*1000, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True) 
    inlier_cloud, _ = inlier_cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=2)
    return inlier_cloud, plane_model
#----------------------------#
# 计算平面点云的最小外接矩形及其姿态信息
# 使用最小外接矩形方法
def plane2obb(inlier_cloud, plane_model, robot_arm, if_soft=False):
    pts = np.asarray(inlier_cloud.points)
    a, b, c , d = plane_model[:]
    n = np.array(plane_model[:3], dtype=float)
    # 保证法向量方向z轴朝下
    if c < 0 :
        n = -n
        d = -d   
    n /= np.linalg.norm(n)
    # 3. 将点投影到平面上： P_proj = P - (n·P + d) * n
    distances = pts.dot(n) + d
    pts_proj = pts - np.outer(distances, n)
    # 4. 以投影质心为原点
    centroid = pts_proj.mean(axis=0)
    pts_rel = pts_proj - centroid
    # 5. 构造平面内正交基 u,v
    #    取一个与 n 不平行的参考向量
    ref = np.array([1,0,0]) if abs(n.dot([1,0,0])) < 0.9 else np.array([0,1,0])
    u = np.cross(ref, n);  u /= np.linalg.norm(u)
    v = np.cross(n, u)
    # 6. 转为 2D 坐标
    pts2d = np.vstack([pts_rel.dot(u), pts_rel.dot(v)]).T.astype(np.float32)
    # 7. 计算最小外接矩形（OpenCV）
    rect = cv2.minAreaRect(pts2d)       # ((cx2, cy2), (w, h), angle)
    (cx2, cy2), (w, h), angle = rect
    box2d = cv2.boxPoints(rect)        # 4个顶点
    # 8. 重映射到 3D 顶点
    rect3d = np.array([centroid + x*u + y*v for x,y in box2d])
    # 9. 计算旋转后的 u 轴（沿长边方向）和 v 轴
    if w < h:
        angle += 90
        w, h = h, w
    ang_rad = np.deg2rad(angle)
    u_rot =  np.cos(ang_rad)*u + np.sin(ang_rad)*v
    v_rot = np.cross(n, u_rot)

    # 10. 构造旋转矩阵 R = [u_rot, v_rot, n]
    R_mat = np.stack([u_rot, v_rot, n], axis=1)
    # 12. 计算矩形中心的3D坐标
    center3d_v = centroid + cx2*u + cy2*v
    if robot_arm == 1:
        offset_distance = 190 # 举例：向上移动137mm , 吸盘高度
    elif robot_arm == 2:
        offset_distance = 190
    if if_soft:
        offset_distance = 180
    center3d = center3d_v - n * offset_distance
    up_distance = 450  # 举例：向上移动20cm
    up3d = center3d - n * up_distance
    stay_distance = 150  # 举例：向上移动20cm
    stay3d = center3d - n * stay_distance
    edges = [
    np.linalg.norm(rect3d[0] - rect3d[1]),
    np.linalg.norm(rect3d[1] - rect3d[2]),
    np.linalg.norm(rect3d[2] - rect3d[3]),
    np.linalg.norm(rect3d[3] - rect3d[0]),
    ]
    edge_lengths = np.array(edges)
    long_edge = max(edge_lengths)
    short_edge = min(edge_lengths)
    return {
        'rect3d':   rect3d,
        'center_v': center3d_v,
        'center':   center3d,
        'R'     :   R_mat,
        'u_axis':   u_rot,
        'v_axis':   v_rot,
        'normal':   n,
        'stay3d':   stay3d,
        'up3d'  :   up3d,
        'long_edge': long_edge,
        'short_edge': short_edge
        }
# 软包算法
### 前置算法
def estimate_curvature(pcd, k=20):
    """估计点云每个点的曲率"""
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(points))
    for i, pt in enumerate(points):
        _, idx, _ = tree.search_knn_vector_3d(pt, k)
        neighbors = points[idx, :]
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-8)
    return curvatures
def region_growing(pcd, min_cluster_size=20, max_dist=0.02*1000, angle_threshold=20, curvature_threshold=0.03):
    """
    区域生长分割（第一个种子为点云中间点序列）

    参数:
        pcd: open3d.geometry.PointCloud
        min_cluster_size: 每个簇最少点数
        max_dist: 邻域点最大距离（米）
        angle_threshold: 法线夹角阈值（度）
        curvature_threshold: 曲率阈值

    返回:
        labels: np.ndarray，每个点的簇标签，-1为噪声
    """
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # 计算曲率
    curvatures = estimate_curvature(pcd, k=20)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    labels = np.full(len(points), -1, dtype=int)
    cluster_id = 0

    # 第一个种子：点云序列中间点
    seed_idx = len(points) // 2
    if curvatures[seed_idx] > curvature_threshold:
        labels[seed_idx] = -1  # 曲率过大，标记为噪声
    else:
        queue = deque([seed_idx])
        labels[seed_idx] = cluster_id
        cluster_size = 1

        while queue:
            current_idx = queue.popleft()
            current_point = points[current_idx]
            current_normal = normals[current_idx]

            _, idx, _ = kdtree.search_radius_vector_3d(current_point, max_dist)
            for neighbor_idx in idx:
                if labels[neighbor_idx] != -1:
                    continue
                neighbor_normal = normals[neighbor_idx]
                angle = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1.0, 1.0))
                if angle < np.deg2rad(angle_threshold) and curvatures[neighbor_idx] < curvature_threshold:
                    labels[neighbor_idx] = cluster_id
                    queue.append(neighbor_idx)
                    cluster_size += 1

        # 小簇标记为噪声
        if cluster_size < min_cluster_size:
            labels[labels == cluster_id] = -1
        else:
            cluster_id += 1

    # 对剩余未标记点执行区域生长（按顺序遍历）
    for i in range(len(points)):
        if labels[i] != -1:
            continue
        if curvatures[i] > curvature_threshold:
            labels[i] = -1
            continue

        queue = deque([i])
        labels[i] = cluster_id
        cluster_size = 1

        while queue:
            current_idx = queue.popleft()
            current_point = points[current_idx]
            current_normal = normals[current_idx]

            _, idx, _ = kdtree.search_radius_vector_3d(current_point, max_dist)
            for neighbor_idx in idx:
                if labels[neighbor_idx] != -1:
                    continue
                neighbor_normal = normals[neighbor_idx]
                angle = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1.0, 1.0))
                if angle < np.deg2rad(angle_threshold) and curvatures[neighbor_idx] < curvature_threshold:
                    labels[neighbor_idx] = cluster_id
                    queue.append(neighbor_idx)
                    cluster_size += 1

        if cluster_size < min_cluster_size:
            labels[labels == cluster_id] = -1
        else:
            cluster_id += 1

    return labels

def soft_obb_info(pcd, robot_arm):
    labels = region_growing(pcd)
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_indices = unique_labels != -1
    if not np.any(valid_indices):
        return None
    cluster_ids = unique_labels[valid_indices]
    cluster_sizes = counts[valid_indices]
    largest_cluster_id = cluster_ids[np.argmax(cluster_sizes)]
    points = np.asarray(pcd.points)
    mask = (labels == largest_cluster_id)
    largest_cluster_points = points[mask]
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    # 计算平面模型
    plane_pcd, plane_model = ransac_plane_cluster(largest_cluster_pcd, distance_threshold=0.005*1000)  # 0.005m
    obb_info = plane2obb(plane_pcd, plane_model, robot_arm = robot_arm, if_soft=True)
    return obb_info
def soft_obb_info_v1(pcd, robot_arm):
    obb = pcd.get_minimal_oriented_bounding_box()
    original_axes = obb.R.copy()  # 3x3 列向量分别为 x, y, z 方向
    x_len, y_len, z_len = obb.extent
    lengths = np.array([x_len, y_len, z_len])
    dot_products = np.array([np.dot(ax, [0,0,1]) for ax in original_axes.T])

    height_idx = np.argmax(np.abs(dot_products))
    n = original_axes[:, height_idx].copy()
    if dot_products[height_idx] < 0:
        n = -n
    h_edge = lengths[height_idx]
    remain_indices = [i for i in range(3) if i != height_idx]
    remain_lengths = lengths[remain_indices]
    remain_axes = original_axes[:, remain_indices]

    sorted_idx = np.argsort(remain_lengths)[::-1]  # 从大到小
    u_rot = remain_axes[:, sorted_idx[0]]
    v_rot = remain_axes[:, sorted_idx[1]]
    u_len = remain_lengths[sorted_idx[0]]
    v_len = remain_lengths[sorted_idx[1]]
    long_edge = u_len
    short_edge = v_len

    if np.dot(np.cross(u_rot, v_rot), n) < 0:
        v_rot = -v_rot
    R_mat = np.column_stack([u_rot, v_rot, n])

    center3d_v = obb.center - 0.5 * h_edge * n
    correction = min(20, h_edge)
    center3d_v = center3d_v + correction * n
    if robot_arm == 1:
        offset_distance = 180 # 举例：向上移动137mm , 吸盘高度
    elif robot_arm == 2:
        offset_distance = 190
    # center3d = center3d_v - n * offset_distance

    # offset_distance = 190  # 吸盘高度偏移
    center3d = center3d_v - n * offset_distance
    up_distance = 450 + correction # 停留距离
    up3d = center3d - n * up_distance
    stay_distance = 150 + correction # 停留距离
    stay3d = center3d - n * stay_distance

    half_u = u_len / 2 * u_rot
    half_v = v_len / 2 * v_rot
    rect3d = np.array([
        center3d_v + half_u + half_v,
        center3d_v + half_u - half_v,
        center3d_v - half_u - half_v,
        center3d_v - half_u + half_v
    ])
    return {
        'rect3d':   rect3d,
        'center_v': center3d_v,
        'center': center3d,
        'R': R_mat,
        'u_axis': u_rot,
        'v_axis': v_rot,
        'normal': n,
        'stay3d': stay3d,
        'up3d'  : up3d,
        'long_edge': long_edge,
        'short_edge': short_edge
    }
def soft_obb_info_v2(pcd, robot_arm):
    labels = region_growing(pcd)
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_indices = unique_labels != -1
    if not np.any(valid_indices):
        return None
    cluster_ids = unique_labels[valid_indices]
    cluster_sizes = counts[valid_indices]
    largest_cluster_id = cluster_ids[np.argmax(cluster_sizes)]
    points = np.asarray(pcd.points)
    mask = (labels == largest_cluster_id)
    largest_cluster_points = points[mask]
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    # 计算平面模型
    plane_pcd, plane_model = ransac_plane_cluster(largest_cluster_pcd, distance_threshold=0.005*1000)  # 0.005m
    obb_info = plane2obb(plane_pcd, plane_model, robot_arm = robot_arm)
    return obb_info

def soft_obb_info_v3(pcd, robot_arm):
    labels = region_growing(pcd)
    unique_labels, counts = np.unique(labels, return_counts=True)   # 这里labels包含了所有的点
    # unique_labels会返回每一种簇的类别并会统计每一种类别的数量存储在counts中
    valid_indices = unique_labels != -1 # return [False, True, True ,...]
    if not np.any(valid_indices):
        return None
    cluster_ids = unique_labels[valid_indices]
    cluster_sizes = counts[valid_indices]
    largest_cluster_id = cluster_ids[np.argmax(cluster_sizes)]  # argument自变量 最大值索引
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    mask = (labels == largest_cluster_id)
    largest_cluster_points = points[mask]
    largest_cluster_normals = normals[mask]
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    largest_cluster_pcd.normals = o3d.utility.Vector3dVector(largest_cluster_normals)
    # 寻找点云面靠近中心的位置
    points = np.asarray(largest_cluster_pcd.points)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    center_point = points[np.argmin(distances)]
    # 利用这一点和这一点的法向量构造一个平面
    normals = np.asarray(largest_cluster_pcd.normals)
    center_normal = normals[np.argmin(distances)]
    center_normal /= np.linalg.norm(center_normal)
    if center_normal[2] < 0:    # 保证法向量z轴朝向大于0
        center_normal = -center_normal
    # 构建二维坐标平面
    z_axis = center_normal 
    arbitrary = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(arbitrary, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    # 将所有点投影到该平面上
    rel = points - center_point
    xs = rel.dot(x_axis)
    ys = rel.dot(y_axis)
    pts2d = np.stack([xs, ys], axis=1)
    #######################################
    # # 法一：最小外接矩形
    # rect = cv2.minAreaRect(pts2d)       # ((cx2, cy2), (w, h), angle)
    # (cx2, cy2), (w, h), angle = rect
    #######################################
    # 法二：PCA
    mean2d = np.mean(pts2d, axis=0)
    cov2d = np.cov((pts2d - mean2d).T)
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    order = np.argsort(eigvals)[::-1]   # arg 返回索引
    eigvecs = eigvecs[:, order]         # 效果是将矩阵列统一修改为由大到小的特征值对应顺序（线性代数中我们关心的是列向量）
    axis2d_x = eigvecs[:, 0]; axis2d_y = eigvecs[:, 1]
    # 目前得到的这两个主成分应当作为抓取姿态的长短边；现在已得到抓取点，接下来主要是可视化的任务
    # 将2D主轴投影到3D
    axis3d_x = axis2d_x[0] * x_axis + axis2d_x[1] * y_axis; axis3d_x /= np.linalg.norm(axis3d_x)
    axis3d_y = np.cross(z_axis, axis3d_x); axis3d_y /= np.linalg.norm(axis3d_y)
    axis3d_z = z_axis
    # 计算抓取矩形，note: 如果这里是外接矩形的话还可以更简单一些；PCA之后，需要计算点在主轴方向上的范围
    coords_in_axes = (pts2d - mean2d).dot(eigvecs)  # 实对称矩阵的不同特征值对应的特征向量必定正交；得到的即是各个点在两个方向上的投影
    min0, max0 = coords_in_axes[:, 0].min(), coords_in_axes[:, 0].max() # 主轴，即长边对应的范围
    min1, max1 = coords_in_axes[:, 1].min(), coords_in_axes[:, 1].max() # 次主轴，即短边对应的范围
    rect_center_2d = mean2d + 0.5 * ((min0 + max0) / 1.0 * eigvecs[:, 0] + (min1 + max1) / 1.0 * eigvecs[:, 1]) # 得到的矩形的2D中心坐标
    corners2d_axes = np.array([[min0, min1],[min0, max1],[max0, max1],[max0, min1],])   # 4x2 
    corners2d = mean2d + corners2d_axes.dot(eigvecs.T)  # eigvecs.T: 2x2 ； .dot 矩阵运算（矩阵、向量矩阵运算；向量与向量内积）；这里实际上是坐标变换至原坐标系
    # 接下来根据得到的各种信息来计算obb_info
    # rect3d 3D坐标系下的矩形四个顶点
    rect3d = center_point[np.newaxis, :] + corners2d[:, 0:1] * x_axis[np.newaxis, :] + corners2d[:, 1:2] * y_axis[np.newaxis, :]
    edges = [np.linalg.norm(rect3d[0] - rect3d[1]), np.linalg.norm(rect3d[1] - rect3d[2]), np.linalg.norm(rect3d[2] - rect3d[3]), np.linalg.norm(rect3d[3] - rect3d[0]), ]
    edge_lengths = np.array(edges)
    # long_edge, short_edge 长边短边，方便后续计算的尺寸异形件
    long_edge = max(edge_lengths); short_edge = min(edge_lengths)
    # center3d_v 所谓可视化下的三维抓取点 实际上是实际的抓取点，为了弥合实际机械爪的长度；目前设置为离质心最近的点而非矩形中心点
    center3d_v = center_point
    # center3d 机械臂抓取中心点
    offset_distance = 190; center3d = center3d_v - center_normal * offset_distance
    # stay3d 机械臂停留点，抓取之前的停留点
    stay_distance = 150; stay3d = center3d - center_normal * stay_distance
    # up3d 机械臂抬升点，抓取之后抬升位置
    up_distance = 450; up3d = center3d - center_normal * up_distance
    # R_mat 旋转矩阵 夹爪最后按照旋转吸取长短边；axis3d_x, axis3d_y 正好为PCA得到的矩形长短边方向
    R_mat = np.column_stack([axis3d_x, axis3d_y, axis3d_z])
    # u_rot  v_rot 为了绘制长短边方向的方向向量
    u_rot = axis3d_x; v_rot = axis3d_y
    # n 法向量，后续会用到法向量剔除
    n = center_normal

    return {
        'rect3d':   rect3d,
        'center_v': center3d_v,
        'center': center3d,
        'R': R_mat,
        'u_axis': u_rot,
        'v_axis': v_rot,
        'normal': n,
        'stay3d': stay3d,
        'up3d'  : up3d,
        'long_edge': long_edge,
        'short_edge': short_edge
        }

def validate_grasp_support(pcd, box, min_points=100*0.8):
    indices = box.get_point_indices_within_bounding_box(pcd.points)
    return len(indices) >= min_points, indices
def adjust_soft_obb(pcd, obb_info, step=10, max_trials=3, width=90, height=20):
    center = obb_info['center_v']
    normal = obb_info['normal']
    R = obb_info['R']
    for i in range(max_trials):
        offset = i * step
        new_center = center + offset * normal
        box = o3d.geometry.OrientedBoundingBox(center=new_center+0.5*normal*height, R=R, extent=[width, width, height])
        valid, indices = validate_grasp_support(pcd, box)
        o3d.visualization.draw_geometries([pcd, box])
        obb_info['center_v'] = new_center
        if valid:
            offset_distance = 210 # 举例：向上移动137mm , 吸盘高度
            center3d = new_center - normal * offset_distance
            obb_info['center'] = center3d
            stay_distance = 200  # 举例：向上移动20cm
            stay3d = center3d - normal * stay_distance
            obb_info['stay3d'] = stay3d
            return obb_info
    return None
## postprocess
def filter_normals(normal):
    # 定义 (0, 0, 1) 方向的向量
    reference_vector = np.array([0, 0, 1])
    # 计算向量的点积
    dot_product = np.dot(normal, reference_vector)
    norm_n = np.linalg.norm(normal)
    norm_reference = np.linalg.norm(reference_vector)
    cos_angle = dot_product / (norm_n * norm_reference)
    return cos_angle

##################################################################
##############################分割线###############################
def visualize_plane2obb_result(pcd, obb_info, show_pcd=True, axis_length=50):
    """
    可视化 plane2obb 返回的结果
    
    pcd:        open3d.geometry.PointCloud - 原始点云
    obb_info:   plane2obb 返回的字典
    show_pcd:   是否显示原始点云
    axis_length:坐标轴长度
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云
    if show_pcd:
        vis.add_geometry(pcd)

    # === 绘制 OBB 框（rect3d 是平面上的 4 个点）===
    rect3d = obb_info['rect3d']
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]  # 四边形的边
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色表示OBB框

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(rect3d),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # === 显示局部坐标系：u, v, n ===
    center_v = obb_info['center_v']  # 原始平面中心
    u_axis = obb_info['u_axis']
    v_axis = obb_info['v_axis']
    n = obb_info['normal']

    axes_points = [
        center_v,
        center_v + u_axis * axis_length * 2,  # u 轴长度加倍
        center_v + v_axis * axis_length,
        center_v + n * axis_length
    ]
    axes_lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    axes_colors = [
        [1, 0, 0],  # u 轴：红色
        [0, 1, 0],  # v 轴：绿色
        [0, 0, 1]   # n 轴：蓝色
    ]

    axes_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines),
    )
    axes_line_set.colors = o3d.utility.Vector3dVector(axes_colors)
    vis.add_geometry(axes_line_set)

    # === 显示抓取中心和停留点 ===
    center = obb_info['center']
    stay3d = obb_info['stay3d']

    # 抓取中心（红色球）
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1, 0, 0])  # 红色
    vis.add_geometry(center_sphere)

    # 停留点（蓝色球）
    stay_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    stay_sphere.translate(stay3d)
    stay_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    vis.add_geometry(stay_sphere)

    # === 显示吸盘方向（从 center 到 center + normal） ===
    arrow_points = [center, center + n * 150]
    arrow_lines = [[0, 1]]
    arrow_colors = [[0, 1, 0]]  # 绿色表示吸盘方向
    arrow_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(arrow_points),
        lines=o3d.utility.Vector2iVector(arrow_lines),
    )
    arrow_line_set.colors = o3d.utility.Vector3dVector(arrow_colors)
    vis.add_geometry(arrow_line_set)

    # === 设置视觉效果 ===
    render_option: o3d.visualization.RenderOption = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0, 0, 0])  # 黑色背景

    view_control: o3d.visualization.ViewControl = vis.get_view_control()
    view_control.set_front([0, -1, -0.5])  # 默认视角
    view_control.set_zoom(0.5)

    vis.run()
    vis.destroy_window()
#-----------------------------------------------------------------#
def show_pointcloud(pcd, colors=None):
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)    
    o3d.visualization.draw_geometries([pcd])
def normal_cluster(pcd, direction=np.array([0, 0, 1]), threshold=0.5, eps=0.02*1000, min_points=5):
    # cos(60°)=0.5 cos(45°)=0.707
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    dot_products = np.sum(normals * direction, axis=1) / (
        np.linalg.norm(normals, axis=1) * np.linalg.norm(direction)
        )
    
    mask = abs(dot_products) > threshold
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    if len(filtered_points) == 0:
        return None

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_pcd
def dbscan_cluster(pcd, eps=0.02, min_samples=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=True))
    max_label = labels.max()
    print(f"Number of clusters: {max_label + 1}")
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 将未聚类成功的点设为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def normal_and_cluster(pcd, direction=np.array([0, 0, 1]), threshold=0.85, eps=0.02*1000, min_points=5):
    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)

    # 法向量估计
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05*1000, max_nn=16)
    )
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # 按照法向量方向过滤点
    dot_products = np.sum(normals * direction, axis=1)
    mask = dot_products > threshold
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    if len(filtered_points) == 0:
        print("筛选后点云为空")
        return None

    # 构造新点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    # 聚类
    labels = np.array(filtered_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()

    if max_label < 0:
        print("聚类失败：没有找到任何有效的簇")
        return None

    # 找最大簇
    cluster_sizes = [(labels == i).sum() for i in range(max_label + 1)]
    largest_cluster_idx = np.argmax(cluster_sizes)
    indices = np.where(labels == largest_cluster_idx)[0]

    # 提取目标点云
    target_pcd = filtered_pcd.select_by_index(indices)

    return target_pcd
# 获取点云的外接长方体框
def extract_obb_info_and_visualize(pcd):
    # NOTE：get_oriented_bounding_box与get_minimal_oriented_bounding_box都可以，前者速度快，后者更准确
    #----------------------------#
    # obb = pcd.get_oriented_bounding_box()
    #----------------------------#
    obb = pcd.get_minimal_oriented_bounding_box()
    obb.color = (1, 0, 0)  # 红色边框
    center = obb.center
    R_matrix = obb.R  # 旋转矩阵
    extent = obb.extent  # 尺寸 [长, 宽, 高]
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axes.translate(center)
    axes.rotate(R_matrix, center=center)
    return {
        "center": center,
        "rotation_matrix": R_matrix,
        "obb": obb
    }


# def depth2pointcloud(x1, y1, depth, intrinsics):
#     # x1, y1: ROI基准坐标
#     height, width = depth.shape
#     x2, y2 = x1 + width, y1 + height
#     parcel_height_img = PointCloudInfo.reference_depth[y1:y2, x1:x2] - depth
#     threshold = 356     # 阈值：机械臂头顶高度
#     num_points_below_threshold = np.sum(parcel_height_img < threshold)
#     # 2pointcloud
#     u, v = np.meshgrid(np.arange(width), np.arange(height))
#     z = depth
#     x = (u + x1 - intrinsics['cx']) * z / intrinsics['fx']
#     y = (v + y1 - intrinsics['cy']) * z / intrinsics['fy']
#     points = np.dstack((x, y, z)).reshape(-1, 3)
#     z = points[:, 2]
#     points = points[(z > 500) & (z < 2000)]
#     z = points[:, 2]
#     # 防止机械臂遮挡，超过阈值的点过高则判断为遮挡，不会计算位姿，放置位姿计算错误
#     ratio = num_points_below_threshold / len(z)
#     if ratio >= 0.1:
#         str = f"z小于{threshold}米的点占比达到{ratio * 100}%"
#         return None, str

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     str = f"z小于{threshold}米的点占比达到{ratio * 100}%"
#     return pcd, str
#####################################################################################
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
