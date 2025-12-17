# main_qt.py
"""
PySide6 版本 YOLO 目标检测 UI
- 使用 DetectionController / FrameSource / Visualizer / DetectionResult 等核心模块
可以使用，AI直接生成，并未使用
"""

import os
import logging
from logging.handlers import RotatingFileHandler

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QSizePolicy,   # ✅ 新增
)

from PySide6.QtGui import QPixmap
from PIL.ImageQt import ImageQt  # 将 PIL.Image 转为 QImage
from app.controller import DetectionController
from core.source import FrameSource, SourceType
from core.visualizer import Visualizer
from core.dto import DetectionResult


# ---------------- 日志初始化（与 Tk 版保持一致） ----------------

def setup_logging():
    """
    初始化全局日志配置：
    - 控制台 + 滚动文件 logs/app.log
    - 级别 INFO（代码内部大量使用 debug，可以按需改成 DEBUG）
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    # 如果已经配置过就不重复配置（防止多次 basicConfig）
    if logging.getLogger().handlers:
        return

    fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(
        level=logging.INFO,  # 若需要更详细日志，可以改为 logging.DEBUG
        handlers=[console_handler, file_handler],
    )


setup_logging()
logger = logging.getLogger(__name__)


class YOLODetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("目标检测软件")
        self.resize(1400, 900)

        # ---------- 状态变量 ----------
        self.model_path = ""                # 模型路径
        self.model_type = "pt"              # 模型格式: pt / onnx

        self.enable_tracking = False        # 是否启用跟踪
        self.speed = 1.0                    # 视频播放速度
        self.save_video = False             # 是否保存检测视频
        self.save_path = ""                 # 视频保存路径
        self.video_writer = None            # cv2.VideoWriter
        self.current_fps = None             # 当前视频 / 摄像头 fps
        self.is_video_mode = False          # 当前是否视频文件模式

        # 控制器 & 帧源
        self.controller = DetectionController()
        self.source = None                  # FrameSource
        self.frame_generator = None         # 帧生成器
        self.is_detecting = False

        # Qt 定时器
        self.result_timer = QTimer(self)
        self.result_timer.timeout.connect(self.poll_results)
        self.result_timer.start(30)         # 30ms 轮询推理结果

        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_step)

        # UI 控件占位
        self.model_path_edit: QLineEdit = None
        self.model_type_combo: QComboBox = None
        self.tracking_checkbox: QCheckBox = None
        self.speed_combo: QComboBox = None
        self.save_video_checkbox: QCheckBox = None

        self.original_label: QLabel = None
        self.result_label: QLabel = None
        self.info_text: QTextEdit = None

        # 缓存当前显示的 pixmap（可选）
        self.original_pixmap: QPixmap = None
        self.result_pixmap: QPixmap = None

        logger.info("YOLODetectorWindow 初始化完成")
        self.setup_ui()

    # ---------------- UI 构建 ----------------

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QGridLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        central_widget.setLayout(main_layout)

        # ========== 模型选择区域 ==========
        model_group = QGroupBox("模型选择", self)
        model_layout = QGridLayout()
        model_group.setLayout(model_layout)

        lbl_model_path = QLabel("模型路径:", self)
        self.model_path_edit = QLineEdit(self)
        btn_select_model = QPushButton("选择模型", self)
        btn_load_model = QPushButton("加载模型", self)

        lbl_model_format = QLabel("模型格式:", self)
        self.model_type_combo = QComboBox(self)
        self.model_type_combo.addItems(["pt", "onnx"])
        self.model_type_combo.setCurrentText("pt")

        model_layout.addWidget(lbl_model_path, 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(btn_select_model, 0, 2)
        model_layout.addWidget(btn_load_model, 0, 3)

        model_layout.addWidget(lbl_model_format, 1, 0)
        model_layout.addWidget(self.model_type_combo, 1, 1)

        # 连接信号
        btn_select_model.clicked.connect(self.select_model)
        btn_load_model.clicked.connect(self.load_model)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)

        # ========== 功能按钮区域 ==========
        func_group = QGroupBox("功能选择", self)
        func_layout = QHBoxLayout()
        func_group.setLayout(func_layout)

        btn_image = QPushButton("图片检测", self)
        btn_camera = QPushButton("摄像头检测", self)
        btn_video = QPushButton("视频检测", self)
        btn_stop = QPushButton("停止检测", self)

        btn_image.clicked.connect(self.detect_image)
        btn_camera.clicked.connect(self.detect_camera)
        btn_video.clicked.connect(self.detect_video)
        btn_stop.clicked.connect(self.stop_detection)

        self.tracking_checkbox = QCheckBox("启用跟踪", self)

        lbl_speed = QLabel("播放速度:", self)
        self.speed_combo = QComboBox(self)
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)

        self.save_video_checkbox = QCheckBox("保存检测视频", self)
        btn_select_save = QPushButton("选择保存路径", self)
        btn_select_save.clicked.connect(self.select_save_path)

        func_layout.addWidget(btn_image)
        func_layout.addWidget(btn_camera)
        func_layout.addWidget(btn_video)
        func_layout.addWidget(btn_stop)
        func_layout.addWidget(self.tracking_checkbox)
        func_layout.addSpacing(15)
        func_layout.addWidget(lbl_speed)
        func_layout.addWidget(self.speed_combo)
        func_layout.addSpacing(15)
        func_layout.addWidget(self.save_video_checkbox)
        func_layout.addWidget(btn_select_save)
        func_layout.addStretch()

        # ========== 显示区域（左右两幅图） ==========
        display_widget = QWidget(self)
        display_layout = QGridLayout()
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setHorizontalSpacing(10)
        display_widget.setLayout(display_layout)

        left_group = QGroupBox("原图", self)
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)

        self.original_label = QLabel("请选择检测功能", self)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid #444;"
        )
        # ✅ 关键：允许 label 比 pixmap 小，不被 pixmap 尺寸“拖着”变大
        self.original_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.original_label.setScaledContents(True)

        left_layout.addWidget(self.original_label)

        right_group = QGroupBox("检测结果", self)
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)

        self.result_label = QLabel("检测结果将在此显示", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid #444;"
        )
        self.result_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.result_label.setScaledContents(True)

        right_layout.addWidget(self.result_label)

        display_layout.addWidget(left_group, 0, 0)
        display_layout.addWidget(right_group, 0, 1)
        display_layout.setColumnStretch(0, 1)
        display_layout.setColumnStretch(1, 1)

        # ========== 信息输出区域 ==========
        info_group = QGroupBox("检测信息", self)
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)

        # ========== 总布局放置 ==========
        main_layout.addWidget(model_group, 0, 0)
        main_layout.addWidget(func_group, 1, 0)
        main_layout.addWidget(display_widget, 2, 0)
        main_layout.addWidget(info_group, 3, 0)

        main_layout.setRowStretch(2, 1)
        main_layout.setRowStretch(3, 0)

        # 简单美化整体背景
        self.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
        }
        """)

    # ---------------- 定时轮询推理结果 ----------------

    def poll_results(self):
        """
        从 DetectionController 的 output_queue 中非阻塞获取结果，
        然后更新左右图和信息文本。
        """
        try:
            result = self.controller.get_result()
            if result:
                original_img, annotated_img, det_result = result
                self.display_image(original_img, self.original_label)
                self.display_image(annotated_img, self.result_label)
                self.display_detection_info(det_result)
                self.maybe_write_video(annotated_img)
        except Exception:
            logger.exception("[UI] 获取结果失败")

    # ---------------- 模型选择 / 加载 ----------------

    def on_model_type_changed(self, text: str):
        self.model_type = text

    def select_model(self):
        current_type = self.model_type
        ext_map = {"pt": "*.pt", "onnx": "*.onnx"}
        filter_str = f"{current_type.upper()} 模型 ({ext_map.get(current_type, '*.*')});;所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {current_type} 模型文件",
            "",
            filter_str,
        )
        if file_path:
            logger.info("选择模型文件: %s", file_path)
            self.model_path = file_path
            self.model_path_edit.setText(file_path)

    def load_model(self):
        try:
            path = self.model_path_edit.text().strip()
            if not path:
                QMessageBox.critical(self, "错误", "请先选择模型文件")
                return
            if not os.path.exists(path):
                QMessageBox.critical(self, "错误", "模型文件不存在")
                return

            logger.info("开始加载模型: %s (type=%s)", path, self.model_type)
            success, info_or_error = self.controller.load_model(path, self.model_type)
            if success:
                info = info_or_error
                task = info.get("task", "")
                device = info.get("device", "")
                model_name = os.path.basename(path)
                logger.info(
                    "模型加载成功: name=%s, task=%s, device=%s",
                    model_name,
                    task,
                    device,
                )
                QMessageBox.information(
                    self,
                    "成功",
                    f"模型加载成功！\n名称: {model_name}\n任务: {task}\n设备: {device}",
                )
            else:
                logger.error("模型加载失败: %s", info_or_error)
                QMessageBox.critical(
                    self,
                    "错误",
                    f"模型加载失败:\n{info_or_error}",
                )
        except Exception:
            logger.exception("模型加载异常")
            QMessageBox.critical(self, "错误", "模型加载失败，请查看日志")

    # ---------------- 保存视频相关 ----------------

    def select_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存视频路径",
            "",
            "MP4 文件 (*.mp4);;AVI 文件 (*.avi);;所有文件 (*.*)",
        )
        if file_path:
            logger.info("选择输出视频路径: %s", file_path)
            self.save_path = file_path

    def init_video_writer_if_needed(self, frame_shape):
        if not self.save_video_checkbox.isChecked():
            return
        if not self.save_path:
            return
        if self.video_writer is not None:
            return

        h, w = frame_shape[:2]

        # 自动选择编码器
        ext = os.path.splitext(self.save_path)[1].lower()
        if ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            codec_name = "MJPG"
        else:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            codec_name = "avc1"

        # FPS
        if self.current_fps is None or self.current_fps <= 0:
            fps = 25.0
        else:
            fps = (
                self.current_fps * max(self.speed, 0.1)
                if self.is_video_mode
                else self.current_fps
            )

        logger.info(
            "初始化视频写入器: path=%s, codec=%s, fps=%.2f, size=(%d,%d)",
            self.save_path,
            codec_name,
            fps,
            w,
            h,
        )

        self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

        if not self.video_writer.isOpened():
            logger.error("无法创建输出视频文件: %s", self.save_path)
            QMessageBox.critical(self, "错误", "无法创建输出视频，请更换保存路径")
            self.video_writer = None

    def maybe_write_video(self, annotated_img):
        if not self.save_video_checkbox.isChecked():
            return
        if not self.save_path:
            return
        if annotated_img is None:
            return

        if self.video_writer is None:
            self.init_video_writer_if_needed(annotated_img.shape)

        if self.video_writer is not None:
            try:
                self.video_writer.write(annotated_img)
            except Exception:
                logger.exception("[SaveVideo] 写入视频帧失败")

    # ---------------- 图片检测 ----------------

    def detect_image(self):
        if self.controller.detector.adapter is None:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)",
        )
        if not file_path:
            return

        try:
            logger.info("开始图片检测: %s", file_path)
            img_source = FrameSource(SourceType.IMAGE, file_path)
            _, img = next(img_source.frames())
            if img is None:
                raise ValueError("无法读取图像")

            result = self.controller.detector.infer(img)
            self.display_image(img, self.original_label)
            self.display_image(result.plot(), self.result_label)
            det_result = DetectionResult.from_yolo(result)
            self.display_detection_info(det_result)
        except Exception as e:
            logger.exception("图片检测失败: %s", e)
            QMessageBox.critical(self, "错误", f"图片检测失败:\n{str(e)}")

    # ---------------- 摄像头 / 视频检测 公共捕获逻辑 ----------------

    def start_capture(self, source_type: SourceType, path: str = None):
        """
        打开帧源并启动推理线程 + 捕获定时器
        """
        # 设置跟踪开关
        self.enable_tracking = self.tracking_checkbox.isChecked()
        self.controller.set_tracking(
            enabled=self.enable_tracking,
            tracker_cfg="bytetrack.yaml",
        )

        # 若需要保存但未设置保存路径，先让用户选
        if self.save_video_checkbox.isChecked() and not self.save_path:
            self.select_save_path()

        # 清理旧 writer
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                logger.exception("关闭旧 VideoWriter 时异常")
            self.video_writer = None

        # 打开帧源
        self.source = FrameSource(source_type, path)
        if not self.source.open():
            if source_type == SourceType.CAMERA:
                logger.error("无法打开摄像头")
                QMessageBox.critical(self, "错误", "无法打开摄像头")
            else:
                logger.error("无法打开视频文件: %s", path)
                QMessageBox.critical(self, "错误", "无法打开视频文件")
            self.source = None
            return False

        self.is_detecting = True
        self.controller.start_inference_thread()
        self.frame_generator = self.source.frames()

        # FPS
        try:
            fps = self.source.cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 25.0
            self.current_fps = fps
        except Exception:
            self.current_fps = 25.0

        if source_type == SourceType.CAMERA:
            logger.info("摄像头 FPS 估计为 %.2f", self.current_fps)
            self.is_video_mode = False
            interval = 30
        else:
            logger.info("视频 FPS 读取为 %.2f", self.current_fps)
            self.is_video_mode = True
            fps = self.current_fps if self.current_fps and self.current_fps > 0 else 25.0
            speed = self.speed if self.speed > 0 else 1.0
            interval = int(1000 / (fps * speed))
            if interval < 1:
                interval = 1

        self.capture_timer.start(interval)
        return True

    def detect_camera(self):
        if self.controller.detector.adapter is None:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        logger.info("开始摄像头检测, 启用跟踪=%s", self.tracking_checkbox.isChecked())
        self.start_capture(SourceType.CAMERA)

    def detect_video(self):
        if self.controller.detector.adapter is None:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)",
        )
        if not file_path:
            return

        logger.info(
            "开始视频检测: %s, 启用跟踪=%s",
            file_path,
            self.tracking_checkbox.isChecked(),
        )

        self.start_capture(SourceType.VIDEO, file_path)

    def capture_step(self):
        """
        Qt 定时器回调：从 frame_generator 拉取一帧送到推理线程
        """
        if not self.is_detecting or self.frame_generator is None:
            return

        try:
            flag, frame = next(self.frame_generator)
        except StopIteration:
            logger.info("帧源 StopIteration 结束")
            self.finish_capture()
            return

        if flag == "end":
            logger.info("帧源结束 (end flag)")
            self.finish_capture()
            if self.is_video_mode:
                QMessageBox.information(self, "完成", "视频播放结束")
            return

        if frame is not None:
            self.controller.submit_frame(frame)

        # 对视频模式，根据当前速度实时调整定时器间隔
        if self.is_video_mode:
            fps = self.current_fps if self.current_fps and self.current_fps > 0 else 25.0
            speed = self.speed if self.speed > 0 else 1.0
            interval = int(1000 / (fps * speed))
            if interval < 1:
                interval = 1
            if self.capture_timer.interval() != interval:
                self.capture_timer.setInterval(interval)

    def finish_capture(self):
        self.is_detecting = False
        self.capture_timer.stop()
        if self.source:
            self.source.release()
            self.source = None
        self.frame_generator = None
        self.current_fps = None
        self.is_video_mode = False

    # ---------------- 停止检测 ----------------

    def stop_detection(self):
        logger.info("收到停止检测请求")
        self.is_detecting = False
        self.capture_timer.stop()
        self.controller.stop_inference_thread()

        if self.source:
            self.source.release()
            self.source = None

        self.frame_generator = None
        self.current_fps = None
        self.is_video_mode = False

        # 关闭视频写入
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                logger.exception("关闭 VideoWriter 时异常")
            self.video_writer = None

        # 重置界面
        self.original_label.setPixmap(QPixmap())
        self.original_label.setText("请选择检测功能")
        self.original_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid #444;"
        )

        self.result_label.setPixmap(QPixmap())
        self.result_label.setText("检测结果将在此显示")
        self.result_label.setStyleSheet(
            "background-color: black; color: white; border: 1px solid #444;"
        )

        self.info_text.clear()
        self.original_pixmap = None
        self.result_pixmap = None

    # ---------------- 显示相关 ----------------

    def display_image(self, cv_img, label: QLabel):
        if cv_img is None:
            return
        try:
            # ✅ 使用固定的最大显示尺寸，避免根据 label 大小不断放大
            max_w, max_h = 800, 600
            pil_img = Visualizer.resize_for_display(cv_img, max_w, max_h)

            qimage = ImageQt(pil_img)  # PIL.Image -> QImage
            pixmap = QPixmap.fromImage(qimage)

            label.setPixmap(pixmap)
            label.setText("")
            label.setStyleSheet("background-color: white; border: 1px solid #ccc;")

            if label is self.original_label:
                self.original_pixmap = pixmap
            elif label is self.result_label:
                self.result_pixmap = pixmap
        except Exception:
            logger.exception("[Display] 显示图像失败")

    def display_detection_info(self, det_result):
        self.info_text.clear()
        try:
            info_text = Visualizer.format_info_text(det_result)
            self.info_text.setPlainText(info_text)
        except Exception:
            logger.exception("[Display] 解析检测结果失败")
            self.info_text.setPlainText("解析结果失败，详细信息请查看日志。\n")

    # ---------------- 其他 ----------------

    def on_speed_changed(self, text: str):
        text = text.replace("x", "")
        try:
            self.speed = float(text)
            logger.info("视频播放速度调整为 %sx", text)
        except ValueError:
            self.speed = 1.0
            logger.warning("播放速度解析失败，重置为 1.0x，原始值: %s", text)

    def closeEvent(self, event):
        """窗口关闭时，确保释放资源"""
        try:
            self.stop_detection()
        except Exception:
            logger.exception("关闭窗口时释放资源异常")
        event.accept()


if __name__ == "__main__":
    logger.info("应用启动")
    app = QApplication([])
    win = YOLODetectorWindow()
    win.show()
    app.exec()
    logger.info("应用正常退出")

