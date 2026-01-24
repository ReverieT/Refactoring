#################################################################
# File: image_process.py
# Part 1: 工程中主要使用的函数
# Part 2: 未使用的函数
#################################################################
__all__ = ['model', 'obb_model',                                # models
           'show_color_image', 'show_depth_image',              # display and visualization
           'depthBit11To8', 'threshTwoPeaks', 'obb_mask',       # process
           'iou_similarity', 'put_text',  # tools
           'compute_depth_diff', 'mask_left', 'mask_left_right',       # for depth

           ]
# models
from ultralytics import YOLO
from utils import path_utils
# model = YOLO('./vision_resources/best.pt')
model = YOLO(path_utils.get_resource_path('resources/vision/best.pt'))
# obb_model = YOLO(path_utils.get_resource_path('resources/vision/best_obb.pt'))
# obb_model = YOLO(path_utils.get_resource_path('resources/vision/weights-11n/last.pt'))
obb_model = YOLO(path_utils.get_resource_path('resources/vision/weights/last.pt'))
# model.predict(source = "./vision_resources/yolo_test.png")
model.predict(source = path_utils.get_resource_path("resources/vision/yolo_test.png"))
obb_model.predict(source = path_utils.get_resource_path("resources/vision/yolo_test.png"))

# image processing
import cv2
import numpy as np

## display and visualization
def show_color_image(color_image, window_name='Color Image'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show_depth_image(depth_image, window_name='Depth Image'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imshow(window_name, depth_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# HACK : 深度图的归一化效果不好，待修改为直接截取

def depthBit11To8(depth_data):
    depth_data_8bit = (depth_data >> 3).astype(np.uint8)  
    return depth_data_8bit

# process
#阈值分割：直方图技术法
def threshTwoPeaks(image):
    # if len(image.shape) == 2:
    #     gray = image
    # else:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    #计算灰度直方图
    #----------------------------#
    def calcGrayHist(grayimage):
        hist = cv2.calcHist([grayimage], [0], None, [256], [0, 256])
        return hist.flatten()
    #----------------------------#
    # def calcGrayHist(grayimage):
    #     rows, cols = grayimage.shape
    #     grayHist = np.zeros([256],np.uint64)
    #     for r in range(rows):
    #         for c in range(cols):
    #             grayHist[grayimage[r][c]] += 1
    #     return grayHist
    #----------------------------#
    histogram = calcGrayHist(gray)
    # NOTE: 对于深度图, 深度图为0无效
    histogram[0] = 0
    maxLoc = np.where(histogram==np.max(histogram))
    firstPeak = maxLoc[0][0]
    measureDists = np.zeros([256],np.float32)
    for k in range(256):
        measureDists[k] = pow(k-firstPeak,2)*histogram[k]
    maxLoc2 = np.where(measureDists==np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    thresh = 0
    if firstPeak > secondPeak:   #第一个峰值在第二个峰值的右侧
        temp = histogram[int(secondPeak):int(firstPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = secondPeak + minloc[0][0] + 1
    else:#第一个峰值在第二个峰值的左侧
        temp = histogram[int(firstPeak):int(secondPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh =firstPeak + minloc[0][0] + 1
    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
    center_depth = gray[center_y, center_x]
    if center_depth < thresh:
        threshImage_out = (gray < thresh).astype(np.uint8) * 255
    else:
        threshImage_out = (gray >= thresh).astype(np.uint8) * 255
    return thresh, threshImage_out
def obb_mask(depth_roi, box):
    """
    生成基于旋转框的mask
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
    xyxyxyxy = box.xyxyxyxy.squeeze(0).cpu().numpy()
    xyxyxyxy = [[int(element) for element in row] for row in xyxyxyxy]
    xyxyxyxy = xyxyxyxy - np.array([x1, y1])
    mask = np.zeros_like(depth_roi, dtype=np.uint8)
    points = xyxyxyxy.reshape((-1, 1, 2))  # 将四个点的坐标转换为 OpenCV 填充所需的格式
    cv2.fillPoly(mask, [points], color=1)  # 在 mask 上绘制旋转框
    return mask
# tools
# 返回两组框标的相似度
#----------------------------#
#----------------------------#
def iou_similarity(box_A, box_B):
    A = np.array(box_A.xyxy.tolist()[0])
    B = np.array(box_B.xyxy.tolist()[0])

    x1 = max(A[0], B[0])
    y1 = max(A[1], B[1])
    x2 = min(A[2], B[2])
    y2 = min(A[3], B[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 并集面积
    area_A = (A[2] - A[0]) * (A[3] - A[1])
    area_B = (B[2] - B[0]) * (B[3] - B[1])
    union_area = area_A + area_B - inter_area

    return inter_area / (union_area + 1e-6)
def put_text(image, label, center_2d, color=(0, 255, 0)):
    cv2.putText(image, label, center_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#----------------------------#

# for depth
depth_ref = cv2.imread(path_utils.get_resource_path('resources/vision/reference_depth.png'), cv2.IMREAD_UNCHANGED)  # uint16
mask_left_path = path_utils.get_resource_path('resources/vision/mask_left.png')
mask_left_right_path = path_utils.get_resource_path('resources/vision/mask_left_right.png')
# if not path_utils.is_file(mask_left_path):
#     raise FileNotFoundError(f"Mask file not found: {mask_left_path}")
# if not path_utils.is_file(mask_left_right_path):
#     raise FileNotFoundError(f"Mask file not found: {mask_left_right_path}")
mask_left = cv2.imread(mask_left_path, cv2.IMREAD_UNCHANGED)
mask_left_right = cv2.imread(mask_left_right_path, cv2.IMREAD_UNCHANGED)
def compute_depth_diff(depth_curr, depth_ref=depth_ref, mask=None, threshold=30, depth_scale=1.0):
    """
    计算掩码区域内两张深度图的差异，如果mask=None，则计算整张图

    参数:
        depth_ref: np.ndarray, uint16 参考深度图
        depth_curr: np.ndarray, uint16 当前深度图
        mask: np.ndarray 或 None, uint8 掩码 (非零=感兴趣区域)，None表示整张图
        threshold: float, 判断大差异的阈值（单位同深度图，默认30mm）
        depth_scale: float, 深度缩放因子（比如RealSense通常是0.001，Kinect一般是1.0）

    返回:
        dict: {
            "mean_diff": 平均差异,
            "max_diff": 最大差异,
            "ratio": 超过阈值的比例
        }
    """

    # 转为float并加上scale
    depth_ref_f = np.squeeze(depth_ref.astype(np.float32) * depth_scale)
    depth_curr_f = np.squeeze(depth_curr.astype(np.float32) * depth_scale)

    diff = np.abs(depth_ref_f - depth_curr_f)


    # 构建有效区域
    valid_mask = (depth_ref_f > 0) & (depth_curr_f > 0)
    
    if mask is not None:
        mask = np.squeeze(mask)   # 保证是 (H, W)
        valid_mask = valid_mask & (mask > 0)


    if np.any(valid_mask):
        diff_masked = diff[valid_mask]
        mean_diff = float(np.mean(diff_masked))
        max_diff = float(np.max(diff_masked))
        ratio = float(np.sum(diff_masked > threshold) / diff_masked.size)
    else:
        mean_diff, max_diff, ratio = np.nan, np.nan, np.nan

    return {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "ratio": ratio
    }














##################################################################
##############################分割线###############################
# 下面的函数并没有在工程中使用
def combined_similarity(box_A, box_B):
    A = box_A.xyxy.tolist()[0]
    B = box_B.xyxy.tolist()[0]
    cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    length_penalty = min(np.linalg.norm(A), np.linalg.norm(B)) / max(np.linalg.norm(A), np.linalg.norm(B))
    return cosine * length_penalty
def get_color_depth_image(depth_image):
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap
def get_mask_from_depth(roi_depth: np.ndarray) -> np.ndarray:
    height, width = roi_depth.shape
    center_x = width // 2
    center_y = height // 2
    center_value = roi_depth[center_y, center_x] #TODO: 需要一些鲁棒性操作
    # 过滤无效值
    valid_mask = roi_depth > 0
    valid_depth = roi_depth[valid_mask]
    # 获得阈值1 高于阈值保留，低于阈值置零
    th1, _ = cv2.threshold(valid_depth, 50, 65535, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    # 获得阈值2
    valid_mask = valid_depth <= th1
    valid_depth = valid_depth[valid_mask]
    th2, _ = cv2.threshold(valid_depth, 50, 65535, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    if th2 > center_value + 10:  #TODO: 需要一些鲁棒性操作
        threshold2 = th2
    else:
        threshold2 = th1
    threshold1 = 2 * center_value - threshold2
    # threshold1 = 2 * center_value - th1
    # 获得两个阈值之间的深度图作为分割mask
    mask = cv2.inRange(roi_depth, threshold1, threshold2)
    return mask

def watershed_segmentation(binary_mask: np.ndarray, color_image: np.ndarray):
    """
    使用分水岭算法对粘连目标进行分割。

    参数：
        binary_mask: 二值图像（前景为255，背景为0）
        color_image: 与 binary_mask 对应的原始彩色图像（3通道）

    返回：
        - markers: 分水岭标签图，每个区域标记不同的整数（边界为 -1）
    """
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # TODO
    # show_color_image(unknown, "Opening")
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)

    markers = markers + 1
    markers[unknown == 255] = 0

    color_with_boundaries = color_image.copy()
    markers = cv2.watershed(color_with_boundaries, markers)
    # 0：unkown  -1:边界  other: 1,2,3,4,5...分割物体连通域
    # TODO
    # - color_with_boundaries: 原始图像上绘制分割边界后的结果
    color_with_boundaries[markers == -1] = [0, 0, 255]
    # show_color_image(color_with_boundaries, "Contours")

    return markers



def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    将掩码 mask 以半透明的颜色叠加在 image 上
    :param image: 原图 (H, W, 3)，BGR格式
    :param mask: 掩码 (H, W)，0或255或bool类型
    :param color: 叠加颜色，BGR格式元组
    :param alpha: 透明度，0~1 之间
    :return: 带掩码叠加的图像
    """
    overlay = image.copy()
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[:, :] = color  # 给整个图填上颜色

    # 使用掩码选中区域进行颜色混合
    mask_binary = (mask > 0)  # 确保是布尔掩码
    overlay[mask_binary] = cv2.addWeighted(image[mask_binary], 1 - alpha, mask_colored[mask_binary], alpha, 0)

    return overlay


def get_grasp_point_and_normal(x1, y1, mask, depth_img, intrinsics, depth_scale=1000.0, patch_size=5):
    """
    输入:
        x1, y1: ROI基准坐标
        mask: 二值图像掩码(np.uint8)
        depth_img: 对应的深度图
        intrinsics: 相机内参字典
        depth_scale: 深度单位缩放 depth_scale==1000 深度单位: mm
        patch_size: 用于估计法向量的邻域大小（像素）

    输出:
        grasp_point_3d: 抓取点的三维坐标 (X, Y, Z)
        normal: 法向量 (nx, ny, nz)
    """
    # Step 1: 计算掩码质心，也可以不转换
    M = cv2.moments(mask.astype(np.uint8))
    # M['m00']：图像的总面积
    if M["m00"] == 0:
        raise ValueError("Empty mask: 无法计算质心")
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Step 2: 获取深度值并反投影为相机坐标
    Z = depth_img[cY, cX] / depth_scale # 单位：m
    if Z == 0:
        raise ValueError("中心深度为0，可能是无效区域")

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    X = (x1 + cX - cx) * Z / fx
    Y = (y1 + cY - cy) * Z / fy
    grasp_point_3d = np.array([X, Y, Z])

    # Step 3: 提取邻域点云用于估计法向量
    h, w = depth_img.shape
    x1 = max(0, cX - patch_size)
    x2 = min(w, cX + patch_size)
    y1 = max(0, cY - patch_size)
    y2 = min(h, cY + patch_size)

    points = []
    for v in range(y1, y2):
        for u in range(x1, x2):
            z = depth_img[v, u] / depth_scale
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
    if len(points) < 3:
        raise ValueError("邻域内有效点不足，无法估计法向量")

    # Step 4: 用 PCA 估计法向量（取主平面法向）
    points_np = np.array(points)
    mean = np.mean(points_np, axis=0)
    cov = np.cov(points_np.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # 最小特征值对应的方向

    # 保证法向量指向摄像头（即Z为负）
    if normal[2] > 0:
        normal = -normal

    return grasp_point_3d, normal