*YOLO 目标检测/跟踪 GUI 工具

一个基于 Ultralytics YOLO 的桌面端目标检测工具，提供：图片检测 / 摄像头实时检测 / 视频检测，可选 多目标跟踪（ByteTrack / BoT-SORT 等），支持检测结果可视化、文本统计，并可选保存检测后的视频。项目采用“UI 与推理解耦”的结构：推理在后台线程运行，界面线程只负责取结果并刷新显示，保证 GUI 流畅。

**功能特性

-支持模型格式：.pt / .onnx（不支持 .pth）

-三种输入源：

--图片文件

--本地视频文件

--摄像头（默认设备 0）

-跟踪模式（可选）：

--开启后使用 model.track(..., persist=True) 获取 track_id

--默认 tracker 配置为 bytetrack.yaml

-视频播放速度（仅视频模式生效）：0.5x / 1.0x / 1.5x / 2.0x

-保存检测视频（可选）：

--.mp4 默认使用 avc1

--.avi 默认使用 MJPG（更通用）

-日志：控制台 + logs/app.log（滚动文件）

**项目结构

-（以当前代码为准，文件名可能因你放置方式略有不同）


  ├── app/
  │   └── controller.py              # DetectionController：协调 UI 与推理线程/队列
  ├── core/
  │   ├── detector.py                # Detector：统一的加载/推理/跟踪接口
  │   ├── dto.py                     # Detection / DetectionResult：结果 DTO
  │   ├── source.py                  # FrameSource：IMAGE/VIDEO/CAMERA 帧源抽象
  │   └── visualizer.py              # Visualizer：缩放显示 + 文本格式化
  ├── infra/
  │   └── ultralytics_adapter.py     # UltralyticsAdapter：封装 YOLO 调用
  ├── main.py                        # Tkinter 版本 GUI（推荐先用这个跑通）
  ├── main_pyside.py                 # PySide6 版本 GUI（如果你要用 Qt）
  ├── requirements.txt
  └── logs/                          # 自动生成（运行后出现）

**环境与安装

-建议 Python 3.8+（能跑 Ultralytics 即可）。

-1) 创建虚拟环境（可选但推荐）
  python -m venv .venv
  #-#-# Win
  .venv\Scripts\activate
  # macOS/Linux
  source .venv/bin/activate

-2) 安装依赖
  pip install -r requirements.txt


-如果你要运行 PySide6 版本界面，还需要额外安装：

  pip install PySide6

*准备模型

-准备任意 Ultralytics YOLO 的模型文件：xxx.pt 或 xxx.onnx。在 GUI 中点击「选择模型」并「加载模型」。

-运行方式
--方式 A：Tkinter 界面（默认/最省事）
---python main.py

--方式 B：PySide6（Qt）界面
---python main_pyside.py

**使用说明（GUI 内）

-加载模型

--选择模型文件（.pt / .onnx）

-点击「加载模型」

--选择检测方式

---「图片检测」：选择一张图片，直接推理并显示结果

---「摄像头检测」：打开摄像头实时检测（可勾选“启用跟踪”）

---「视频检测」：选择视频文件逐帧检测（可调播放速度）

-启用跟踪（可选）

-勾选「启用跟踪」

-摄像头/视频模式会走 track 流程，并显示跟踪 ID（如模型/任务支持）

-保存检测视频（可选）

-勾选「保存检测视频」

-点击「选择保存路径」

-在摄像头/视频检测过程中，会把检测后的帧写入文件

-结果展示说明

--左侧：原始帧

--右侧：绘制框后的结果帧（result.plot()）

--下方文本框：逐目标信息 + 分类统计（类别名、置信度、bbox 坐标；开启跟踪时显示 track_id；分割任务时可显示“有分割掩码”和掩码像素面积，如可用）

*设计要点（简述）

-DetectionController 维护输入/输出队列（maxsize=1），并采用“丢弃旧帧保留最新帧”的策略，保证实时性。

-推理线程：关闭跟踪走 Detector.infer()；开启跟踪走 Detector.track(..., persist=True, tracker=xxx)。

-DetectionResult.from_yolo() 将 Ultralytics 的结果转换为本地 DTO，便于 UI 层格式化输出与扩展（分割/跟踪等）。

*常见问题
-1) 选择了模型但推理无反应？

--确认已点击「加载模型」，并弹出加载成功提示。

--查看 logs/app.log 是否有报错信息。

-2) ONNX 模型无法切到 GPU？

--部分 ONNX 在 .to('cuda') 时会失败，代码会自动回退到 CPU，这是正常保护逻辑。

-3) 摄像头打不开？

--检查是否被其他软件占用

--尝试修改摄像头设备 ID（代码里默认 0）
