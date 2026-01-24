# PostalDAS_Vision

## 1. 系统架构与可视化目录树

```
# 项目根目录
├── vision.py                  # 原VisionModule.py，改名规避歧义
├── decision.py                # 对外暴露的决策门面类（原DecisionModule.py）
├── readme.md                       # 项目说明
├── requirements.txt                # 项目依赖（可选）
├── vision_module/                  # 视觉模块内部文件夹（存放内部实现，不对外暴露）
│   ├── __init__.py                 # 空文件，标记为Python包
│   ├── config.py                   # 视觉模块配置（相机、包裹、算法参数）
│   ├── camera_driver.py            # 相机驱动类（封装相机启停、取图线程、ImageFrame入队）      [测试通过✅]
│   ├── data_structures.py          # 数据结构类（定义ImageFrame、后续可扩展ParcelInfo等）
│   ├── region_manager.py           # 区域管理类（定义包裹区域、后续可扩展其他区域等）
│   ├── vision_utils.py             # 视觉模块级工具（可选，阶段一可暂不创建）
│   ├── vision_process/             # 视觉底层工具子模块（图像处理、点云处理，无业务耦合）
│   │   ├── __init__.py             # 空文件，标记为Python子包
│   │   ├── image_processing.py     # 2D图像底层处理（去噪、格式转换、绘制等）
│   │   └── point_cloud_processing.py # 3D点云底层处理（下采样、法向量、RANSAC等）
│   └── vision_tools/               # 视觉通用工具子模块（无业务耦合，可选）
│       ├── __init__.py             # 空文件，标记为Python子包
│       └── image_utils.py          # 额外视觉工具（可选，阶段一可暂不创建）
└── decision_module/                # 决策模块内部文件夹（存放内部实现，不对外暴露）
    ├── __init__.py                 # 空文件，标记为Python包
    ├── config.py                   # 决策模块配置（状态、延时、机械臂参数）
    ├── state_helpers.py            # 系统状态辅助实现（迁移原DecisionModule的状态方法）
    └── decision_utils.py           # 决策通用工具（队列清理、机械臂等待等）
```

