# YOLO 目标检测/跟踪 GUI 工具

这是一个基于 **Ultralytics YOLO** 的桌面端目标检测与跟踪工具。

项目采用 **“UI 与推理解耦”** 的架构设计：推理任务在后台独立线程运行，界面线程仅负责获取结果并刷新显示，从而保证在进行重型模型推理时 GUI 依然保持流畅响应。

## ✨ 功能特性

* **多模型支持**：支持加载 `.pt` 和 `.onnx` 格式模型（Ultralytics 导出）。
* **多输入源**：
    * 🖼️ **图片文件**：单张图片推理。
    * 📹 **本地视频文件**：支持倍速播放 (0.5x - 2.0x)。
    * 📷 **摄像头实时检测**：默认调用设备 0。
* **目标跟踪 (Tracking)**：
    * 可选开启/关闭。
    * 集成 `ByteTrack` / `BoT-SORT` (默认配置为 bytetrack.yaml)。
    * 支持 `model.track(..., persist=True)` 保持 ID 连续性。
* **结果可视化**：
    * 左侧显示原始帧，右侧显示绘制检测框/掩码后的结果帧。
    * 实时文本统计：类别、置信度、BBox 坐标、Track ID。
* **视频录制**：
    * 支持将检测后的画面保存为视频文件。
    * `.mp4` (avc1) 或 `.avi` (MJPG) 格式。
* **日志系统**：控制台输出 + `logs/app.log` 滚动记录。

## 📂 项目结构

```text
.
├── app/
│   └── controller.py              # DetectionController：协调 UI 与推理线程/队列
├── core/
│   ├── detector.py                # Detector：统一的加载/推理/跟踪接口
│   ├── dto.py                     # DetectionResult：结果 DTO转换
│   ├── source.py                  # FrameSource：帧源抽象 (IMAGE/VIDEO/CAMERA)
│   └── visualizer.py              # Visualizer：图像绘制与文本格式化
├── infra/
│   └── ultralytics_adapter.py     # UltralyticsAdapter：封装 YOLO 调用
├── main.py                        # Tkinter 版本 GUI入口（推荐）
├── main_pyside.py                 # PySide6 版本 GUI入口（可选）
├── requirements.txt               # 项目依赖
└── logs/                          # 运行时自动生成
