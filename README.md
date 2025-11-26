# YOLO GUI 目标检测工具

🎯 YOLO GUI 目标检测工具

一个基于 Ultralytics YOLO 的本地桌面图形化目标检测工具，内置模型加载、图片/视频/摄像头检测、多目标跟踪、视频保存和日志系统。

简洁易用、结构清晰，适合作为 本地检测工具 / 二次开发框架 / 模型测试平台。

🚀 功能特性
🟢 检测能力

图片检测（JPG / PNG / BMP）

视频检测（MP4 / AVI / MOV）

摄像头实时检测

支持 YOLO 的所有任务类型：

检测 (detect)

分割 (segment)

姿态 (pose)

旋转框 (obb)

(取决于模型本身)

🟢 模型兼容

支持 Ultralytics YOLO 原生格式：

*.pt

*.onnx

自动识别模型任务类型（detect/seg/pose/obb）

自动选择 GPU / CPU 设备

🟢 多目标跟踪（可选）

内置 Ultralytics YOLO 的 track 模式

默认支持 ByteTrack（可扩展 BOTSORT）

在检测结果中显示：

Track ID

目标框

分割区域（若为 seg 模型）

可持久化 track (persist=True)

🟢 视频保存

可选保存检测结果视频（摄像头 & 视频均支持）

自动选择最兼容的编码器：

MP4 → avc1（H.264）

AVI → MJPG

支持自定义保存路径

🟢 视频播放速度控制

在视频检测模式中可调节播放速度：

0.5x

1.0x

1.5x

2.0x

视频播放使用真实 FPS 和倍速计算，播放更加顺畅。

🟢 完整日志系统（专业级）

控制台仅显示 INFO（简洁）

logs/app.log 保存 DEBUG 全量日志（排查问题用）

自动日志滚动（最大 5MB，保留 3 个历史文件）

记录内容包括：

模型加载

视频/摄像头打开与释放

推理线程状态

每帧推理过程

跟踪相关信息

视频保存状态与编码器

错误/异常堆栈

🧱 项目结构

建议使用如下目录结构：

your_project/
  main.py                  # GUI + 日志初始化（实际运行入口）
  logs/
    app.log                # 程序运行日志（自动生成）
  app/
    __init__.py
    controller.py          # 检测控制器（管理后台推理线程）
  core/
    __init__.py
    detector.py            # 模型加载/推理/跟踪统一入口
    dto.py                 # Detection / DetectionResult 数据结构
    source.py              # FrameSource（图片/视频/摄像头）
    visualizer.py          # 图像缩放 + 输出文本格式化
  infra/
    __init__.py
    ultralytics_adapter.py # 对接 Ultralytics YOLO
  models/                  # (可选) 模型文件目录
    yolov8s.pt
  trackers/                # (可选) 跟踪器配置文件
    bytetrack.yaml

🔧 环境依赖
Python 版本

推荐：

Python 3.8 - 3.10

安装依赖
pip install ultralytics opencv-python pillow torch torchvision


如需 GPU，请按 PyTorch 官方文档安装对应 CUDA 版本。

💻 运行方法

在项目根目录运行：

python main.py


即可打开完整 GUI 界面。

🧭 使用说明
1️⃣ 加载模型

点击「选择模型」

选择 .pt 或 .onnx 文件

确认模型格式下拉框（pt / onnx）

点击「加载模型」

成功后会弹窗提示 + 日志记录模型任务类型与运行设备。

2️⃣ 图片检测

点击「图片检测」

选择一张图片

原图和检测结果分别显示在左/右面板

下方显示：

目标类别

置信度

坐标

分割掩码面积（若有）

track ID（若启用跟踪）

3️⃣ 摄像头检测

加载模型

可选勾选：
-「启用跟踪」
-「保存检测视频」

点击「摄像头检测」

检测过程中可实时停止检测。

4️⃣ 视频检测

加载模型

可选勾选跟踪 / 保存视频

可调节播放速度

点击「视频检测」并选择视频文件

播放结束会自动提示「视频播放结束」。

📝 日志系统说明

所有运行日志默认输入到：

logs/app.log


日志级别说明：

输出位置	级别	用途
控制台	INFO+	显示关键事件
logs/app.log	DEBUG ~ CRITICAL	调试排错、详细跟踪

日志内容包括：

应用启动/退出

模型加载成功/失败

Ultralytics 模型详情（task / classes / device）

视频/摄像头运行状态

推理线程·队列状态（丢帧）

多目标跟踪相关 ID 信息

分割掩码面积

视频保存参数（fps / codec / size）

错误与异常堆栈信息

❓ 常见问题 (FAQ)
❓ 视频保存后无法在 Windows 播放？

本程序已自动适配 Windows 播放器：

MP4 → avc1（H.264）

AVI → MJPG

若仍无法播放，请：

升级 OpenCV（需带 ffmpeg）

使用 VLC / PotPlayer 等播放器

❓ Track ID 不显示？

确保：

已勾选「启用跟踪」

模型支持检测任务（detect/seg/obb 都支持跟踪）

tracker 配置文件正确（默认使用 ByteTrack）

❓ 摄像头卡顿？

队列采用“始终保留最新帧”的设计

推理线程和 GUI 主线程分离

控制台 DEBUG 日志可查看丢帧情况

🛠 可扩展方向（建议）

 增加自动 FPS 计算显示

 增加截图按钮（保存检测帧）

 自定义 tracker 配置文件选择

 增加模型管理器（自动显示模型任务类型）

 增加批量图片检测功能


