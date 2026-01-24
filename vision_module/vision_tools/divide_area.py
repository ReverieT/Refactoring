import cv2
import numpy as np
# 读取图像
# color_data = cv2.imread("static/vision/datasets/color20250610_11-06-40_824302.png")
color_data = cv2.imread("resources/vision/color_test.png")


img = color_data
height, width = img.shape[:2]

# 创建一张空图用于绘制区域可视化
overlay = np.zeros_like(img)
# 初始化掩码
mask_left_right = np.zeros((height, width), dtype=np.uint8)  # 左+右
mask_left = np.zeros((height, width), dtype=np.uint8)        # 左区

# 区域可视化
for y in range(height):
    for x in range(width):
        # 死区 >0 >0 >0
        # L1最左侧线 L2上测线 L6右区斜上测线
        L1 = -727*x + -8*y + 80267
        L2 = 8*x + -627*y + 81241
        L6 = 170*x + -557*y + -21269

        # 左区 < 0
        # L3 左区右侧线
        L3 = 444*x + 3*y + -253239
        # L3 = 882*x + -103*y + -391712

        # 右侧 >0 >0    
        # L4 右区左侧线   L5 右区下侧线
        L4 = 778*x + 11*y + -564045
        # L4 = 670*x + 14*y + -548320
        L5 = 270*x + -601*y + 234088
        if (not(L1 > 0 or L2 > 0 or L6 > 0) and (L3 <0)):
            overlay[y, x] = [0, 255, 0]
            mask_left_right[y, x] = 1
            mask_left[y, x] = 1
        elif (not(L1 > 0 or L2 > 0 or L6 > 0) and (L4 >0 and L5 > 0)):
            overlay[y, x] = [0, 0, 255]
            mask_left_right[y, x] = 1
        else:
            overlay[y, x] = [255, 255, 255]

result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
# 保存或显示掩码
# TODO: 确定掩码保存在哪里
cv2.imwrite("resources/vision/mask_left_right.png", mask_left_right * 255)
cv2.imwrite("resources/vision/mask_left.png", mask_left * 255)

# 保存点击点
clicked_points = []

# 鼠标点击回调函数
def mouse_callback(event, x, y, flags, param):
    global result, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        img = result.copy()
        clicked_points.append((x, y))
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        # 如果已经点击两个点
        if len(clicked_points) == 2:
            (x1, y1), (x2, y2) = clicked_points

            # 计算直线参数 A, B, C 使得 Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2

            # 避免除0错误并格式化输出
            if B != 0:
                k = -A / B
                b = -C / B
                equation = f"y = {k:.3f}x + {b:.1f}"
            else:
                equation = f"x = {x1}"

            print(f"直线方程：{A}*x + {B}*y + {C} = 0   =>   {equation}")


            # 绘制直线
            x_vals = np.array([0, width])
            if B != 0:
                y_vals = (-A * x_vals - C) / B
                pts = np.array([[x_vals[0], int(y_vals[0])], [x_vals[1], int(y_vals[1])]], np.int32)
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)
            else:
                cv2.line(img, (x1, 0), (x1, height), (0, 255, 255), 2)

            # 显示直线方程
            cv2.putText(img, equation, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            clicked_points = []  # 重置

        cv2.imshow("Partitioned Image", img)

# 设置窗口与鼠标回调
cv2.namedWindow("Partitioned Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Partitioned Image", mouse_callback)
cv2.imshow("Partitioned Image", result)

# 主循环，仅支持 q 退出
while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
