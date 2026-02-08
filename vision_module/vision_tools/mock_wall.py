import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import sys

class PlaneFittingTool:
    def __init__(self, pcd):
        self.pcd = pcd
        self.picked_points = []
        self.marker_spheres = []
        self.plane_name = "fitted_plane" 

        # --- 关键修改 1: 预先构建 KDTree，防止点击时构建导致的卡顿或崩溃 ---
        print("Building KDTree for point cloud...")
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        print("KDTree built.")

        # 1. 初始化 GUI
        try:
            gui.Application.instance.initialize()
        except Exception as e:
            pass # 如果已经初始化过则忽略

        self.window = gui.Application.instance.create_window("Plane Fitting Tool v2.1 (Stable)", 1280, 720)
        
        # 2. 创建 3D 场景
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        # ===【新增这一行】设置鼠标控制模式为旋转 ===
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        
        mat = rendering.MaterialRecord()
        mat.point_size = 4.0 
        mat.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("pcd", self.pcd, mat)
        
        # 3. 侧边栏
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        
        label = gui.Label("Instructions:\n1. Hold [Shift + Left Click] to Pick\n2. Pick >= 3 points\n3. Click 'Fit Plane'")
        self.panel.add_child(label)

        btn_fit = gui.Button("Fit Plane")
        btn_fit.set_on_clicked(self.on_fit)
        self.panel.add_child(btn_fit)

        btn_reset = gui.Button("Reset / Clear")
        btn_reset.set_on_clicked(self.on_reset)
        self.panel.add_child(btn_reset)
        
        btn_print = gui.Button("Print Params")
        btn_print.set_on_clicked(self.on_print_params)
        self.panel.add_child(btn_print)

        self.window.set_on_layout(self.on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        self.scene_widget.set_on_mouse(self.on_mouse_pick)

    def on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 20 * layout_context.theme.font_size 
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.get_right() - panel_width, r.height)

    def on_mouse_pick(self, event):
        # 检查 Shift 键
        is_shift = (event.modifiers & int(gui.KeyModifier.SHIFT))
        
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and is_shift:
            # 记录点击位置，供 callback 使用
            # 注意：必须考虑 DPI 缩放，但在 SceneWidget 内部相对坐标通常是像素
            mouse_x = event.x - self.scene_widget.frame.x
            mouse_y = event.y - self.scene_widget.frame.y
            
            def post_pick(depth_image):
                try:
                    # --- 关键修改 2: 安全的数据转换 ---
                    # Open3D 的 Image 对象不能直接下标访问，必须转 numpy
                    depth_array = np.asarray(depth_image)
                    
                    # 检查坐标是否越界 (防崩溃)
                    h, w = depth_array.shape
                    x = int(mouse_x)
                    y = int(mouse_y)
                    
                    if x < 0 or x >= w or y < 0 or y >= h:
                        print(f"Click out of bounds: ({x}, {y}) vs ({w}, {h})")
                        return

                    # 获取深度值
                    d = depth_array[y, x]
                    
                    # 检查无效深度 (比如背景)
                    if d <= 0 or not np.isfinite(d):
                        print("Clicked on background (invalid depth)")
                        return

                    # 反投影
                    world_pos = self.scene_widget.scene.camera.unproject(
                        x, y, d, self.scene_widget.frame.width, self.scene_widget.frame.height
                    )
                    
                    # --- 关键修改 3: 使用预构建的 Tree 查找最近点 ---
                    # search_knn_vector_3d 返回 [k, indices, distances]
                    _, idx, _ = self.pcd_tree.search_knn_vector_3d(world_pos, 1)
                    picked_pt = np.asarray(self.pcd.points)[idx[0]]
                    
                    # 只有在主线程更新 UI 比较安全，虽然这里也可以，但建议尽量简单
                    self.add_picked_point(picked_pt)
                    
                except Exception as e:
                    print(f"Error in post_pick: {e}")
                    import traceback
                    traceback.print_exc()

            self.scene_widget.scene.scene.render_to_depth_image(post_pick)
            return gui.Widget.EventCallbackResult.HANDLED
            
        return gui.Widget.EventCallbackResult.IGNORED

    def add_picked_point(self, point):
        # 将 UI 更新逻辑分离出来
        self.picked_points.append(point)
        print(f"Picked: {point}")
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])
        
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        
        name = f"sphere_{len(self.picked_points)}"
        self.scene_widget.scene.add_geometry(name, sphere, mat)
        self.marker_spheres.append(name)
        
        self.window.post_redraw()

    def on_fit(self):
        if len(self.picked_points) < 3:
            print("Error: Need at least 3 points.")
            return
        
        pts = np.array(self.picked_points)
        centroid = np.mean(pts, axis=0)
        
        # SVD
        try:
            _, _, vh = np.linalg.svd(pts - centroid)
            normal = vh[-1, :]
            A, B, C = normal
            D = -np.dot(normal, centroid)
            
            self.current_plane_params = (A, B, C, D)
            print(f"\n[Fitted Plane] {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
            
            if self.scene_widget.scene.has_geometry(self.plane_name):
                self.scene_widget.scene.remove_geometry(self.plane_name)
            
            mesh = self.create_plane_mesh(A, B, C, D, centroid)
            
            mat = rendering.MaterialRecord()
            mat.base_color = [0.1, 0.1, 0.9, 0.5]
            mat.shader = "defaultLitTransparency"
            # 必须开启双面渲染，否则从背面看是透明的
            # mat.sRGB_color_space = True 
            
            self.scene_widget.scene.add_geometry(self.plane_name, mesh, mat)
            self.window.post_redraw()
            
        except Exception as e:
            print(f"Fit failed: {e}")

    def create_plane_mesh(self, A, B, C, D, center, size=0.4):
        n = np.array([A, B, C])
        if abs(A) < 0.9:
            u = np.cross(n, [1, 0, 0])
        else:
            u = np.cross(n, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        p1 = center + size*u + size*v
        p2 = center + size*u - size*v
        p3 = center - size*u - size*v
        p4 = center - size*u + size*v
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([p1, p2, p3, p4])
        # 增加反向三角形，确保双面可见
        mesh.triangles = o3d.utility.Vector3iVector([
            [0, 1, 2], [0, 2, 3], # 正面
            [2, 1, 0], [3, 2, 0]  # 背面
        ])
        mesh.compute_vertex_normals()
        return mesh

    def on_reset(self):
        print("Resetting...")
        for name in self.marker_spheres:
            self.scene_widget.scene.remove_geometry(name)
        
        if self.scene_widget.scene.has_geometry(self.plane_name):
            self.scene_widget.scene.remove_geometry(self.plane_name)
            
        self.marker_spheres = []
        self.picked_points = []
        self.window.post_redraw()

    def on_print_params(self):
        if hasattr(self, 'current_plane_params'):
            A, B, C, D = self.current_plane_params
            print("-" * 30)
            print(f"Plane Equation: {A:.5f}x + {B:.5f}y + {C:.5f}z + {D:.5f} = 0")
            print("-" * 30)
        else:
            print("No plane fitted yet.")

    def run(self):
        gui.Application.instance.run()

def generate_point_cloud_from_images(rgb_path, depth_path, fx, fy, cx, cy):
    # 这里保持不变
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    width = np.asarray(color_raw).shape[1]
    height = np.asarray(color_raw).shape[0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

if __name__ == "__main__":
    RGB_FILE = "resources/vision/color.png"
    DEPTH_FILE = "resources/vision/depth.png"
    MY_FX, MY_FY = 981.4576, 981.4964
    MY_CX, MY_CY = 706.5240, 575.2091
    
    if not os.path.exists(RGB_FILE) or not os.path.exists(DEPTH_FILE):
        print("Error: Images not found! Please check path.")
    else:
        pcd = generate_point_cloud_from_images(RGB_FILE, DEPTH_FILE, MY_FX, MY_FY, MY_CX, MY_CY)
        
        # 简单过滤，防止有太远的离群点影响视角
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        if len(pcd.points) == 0:
            print("Error: Point cloud is empty!")
        else:
            app = PlaneFittingTool(pcd)
            app.run()



"""
记录参数（单位：m）：
    1. 左区左侧挡板方程：
    Plane Equation: 0.97063x + -0.00059y + 0.24058z + 1.01269 = 0
    2. 左区上侧挡板方程：
    Plane Equation: -0.00962x + -0.99966y + 0.02403z + 0.44453 = 0
    3. 右区挡板方程：
    Plane Equation: 0.30123x + 0.95335y + -0.01980z + 0.30762 = 0
"""
# 挡板区域定义：（单位：mm）
# 挡板所有规定均为平面法向量朝向抓取区域
LEFT_PLANES = [
    # 左区左侧挡板
    np.array([0.97063, -0.00059, 0.24058, 1012.69]),
    # 左区上侧挡板
    np.array([0.00962, 0.99966, -0.02403, -444.53]),
]
# 左侧挡板高度（单位：mm）
LEFT_PLANE_HEIGHT = 1012.69 #[TODO]: 需要进行修改
RIGHT_PLANES = [
    np.array([-0.30123, -0.95335, 0.01980, -307.62]),
]
RIGHT_PLANE_HEIGHT = 307.62 #[TODO]: 需要进行修改
# 安全阈值（单位：mm）
SAFETY_THRESHOLD = 50.0     # 5cm
# 工具函数
def point2plane_distance(point, plane):
    """
    计算点到平面的距离
    """
    A, B, C, D = plane
    x, y, z = point
    return abs(A*x + B*y + C*z + D) / np.sqrt(A**2 + B**2 + C**2)
def security_wall(point, normal, arm_id):
    """
    安全墙，防止机械臂与挡板进行碰撞，左区挡板方程有二；右区挡板方程有一
    首先判断包裹抓取点（center_v）与挡板方程之间的距离，需要大于安全阈值
    其次需要判断法向量不能在挡板外侧，导致抓取时撞击挡板
    * Parameter:
        point: 包裹抓取点（center_v）
        normal: 包裹抓取点的法向量（注：已确保法向量方向z轴超下）
        arm_id: 机械臂ID，1为左区，2为右区
    """
    if arm_id == 1:
        planes = LEFT_PLANES
        height = LEFT_PLANE_HEIGHT
    elif arm_id == 2:
        planes = RIGHT_PLANES
        height = RIGHT_PLANE_HEIGHT

    for index, plane in enumerate(planes):
        # 一、距离检查: 距离挡板很近且抓取点在挡板之下
        distance = point2plane_distance(point, plane)
        if distance < SAFETY_THRESHOLD:
            if point[2] > height:
                return False
        # 二、法向量检查: 法向量不能在挡板外侧
        if distance < SAFETY_THRESHOLD*2:
            A, B, C, D = plane
            normal_dot_plane = np.dot(normal, np.array([A, B, C]))
            if normal_dot_plane > 0.5:    # 说明法向量夹角是钝角[TODO]:还需要设置一个特定阈值，修改点积为cos值;或提前归一化好两个法向量
                # 可以根据现场挡板高度来计算出特定的阈值
                return False
    return True
