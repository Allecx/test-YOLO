# YOLO GUI 目标检测工具

一个基于 **Ultralytics YOLO** 的本地桌面图形化目标检测工具，支持模型加载、图片/视频/摄像头检测、多目标跟踪、视频保存、分割支持，以及专业级日志系统。

适合作为 **模型测试平台 / 轻量级检测软件 / 工程化前端框架**。

---

## ✨ 功能特性

### 🟢 检测功能
- 图片检测（JPG / PNG / BMP）
- 视频检测（MP4 / AVI / MOV 等）
- 摄像头实时检测
- 自动识别 YOLO 模型任务类型：
  - 检测 (detect)
  - 分割 (segment)
  - 姿态 (pose)
  - 旋转框 (obb)

---

### 🟢 模型支持
- Ultralytics YOLO `*.pt`
- YOLO ONNX 格式 `*.onnx`
- 自动选择 GPU / CPU
- 显示模型任务类型和类别数

---

### 🟢 多目标跟踪（可选）
- 使用 `model.track()` 原生接口
- 默认使用 **ByteTrack**
- 支持 ID 持续跟踪（persist=True）
- 显示：
  - Track ID  
  - 目标框  
  - 分割掩码（若模型支持）  

---

### 🟢 视频保存
- 可保存带检测框 / 掩码 / track ID 的结果视频
- 自动选择兼容编码器：
  - MP4 → `avc1`（H.264）
  - AVI → `MJPG`
- 摄像头 & 视频模式均支持

---

### 🟢 视频播放速度调节
- 视频检测时可选：
  - `0.5x`
  - `1.0x`
  - `1.5x`
  - `2.0x`
- 按真实 FPS × 倍速播放

---

### 🟢 日志系统（专业级）
- 控制台输出：`INFO`（简洁）
- 日志文件：`logs/app.log`，记录全部 `DEBUG` 日志
- 自动滚动：每文件 5MB，保留 3 个
- 记录内容包括：
  - 应用启动/退出
  - 模型加载成功/失败
  - 推理线程状态
  - 视频/摄像头帧源状态
  - 跟踪信息（track_id）
  - 分割掩码面积
  - 视频保存信息
  - 异常堆栈

---

## 📁 项目结构

```text
your_project/
  main.py                  # GUI + 日志初始化（入口）
  logs/
    app.log                # 日志文件（自动生成）
  app/
    controller.py          # 控制推理线程、队列、跟踪开关
  core/
    detector.py            # 模型加载 / 推理 / 跟踪
    dto.py                 # 数据结构定义（DetectionResult）
    source.py              # 图像/视频/摄像头帧源
    visualizer.py          # 图像缩放 + 文本格式化
  infra/
    ultralytics_adapter.py # Ultralytics YOLO 接口封装
  models/                  # (可选) 模型文件目录
  trackers/                # (可选) 跟踪器配置，如 bytetrack.yaml



