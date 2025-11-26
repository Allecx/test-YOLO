# main.py  / 或 gui/app.py（根据你当前文件名放置）

import os
import logging
from logging.handlers import RotatingFileHandler

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import ImageTk

from app.controller import DetectionController
from core.source import FrameSource, SourceType
from core.visualizer import Visualizer
from core.dto import DetectionResult


# ---------------- 日志初始化 ----------------

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
        level=logging.INFO,  # 如果想更详细的日志改成 logging.DEBUG
        handlers=[console_handler, file_handler],
    )


# 先初始化日志，再导入/使用其他模块
setup_logging()
logger = logging.getLogger(__name__)


class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("目标检测软件")
        self.root.geometry("1400x900")

        # 基本状态
        self.model_path = tk.StringVar()
        self.model_type = tk.StringVar(value="pt")  # pt / onnx

        # 跟踪开关
        self.enable_tracking_var = tk.BooleanVar(value=False)

        # 播放速度（只对视频检测生效）
        self.speed_var = tk.DoubleVar(value=1.0)  # 0.5, 1.0, 1.5, 2.0

        # 保存检测视频选项
        self.save_video_var = tk.BooleanVar(value=False)
        self.save_path = tk.StringVar(value="")     # 输出视频路径
        self.video_writer = None                    # cv2.VideoWriter
        self.current_fps = None                     # 当前会话的 FPS（视频读取或摄像头）
        self.is_video_mode = False                  # 当前是否在视频检测模式

        # 控制器 & 帧源
        self.controller = DetectionController()
        self.source = None
        self.frame_generator = None
        self.is_detecting = False

        # Tk 图片引用，防止被 GC
        self.original_tk_image = None
        self.result_tk_image = None

        logger.info("YOLODetectorApp 初始化完成")
        self.setup_ui()
        self.poll_results()

    # ---------------- UI 构建 ----------------

    def setup_ui(self):
        """构建用户界面（布局保持不变）"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ========== 模型选择区域 ==========
        model_frame = ttk.LabelFrame(main_frame, text="模型选择", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(
            row=0, column=1, padx=5, sticky=(tk.W, tk.E)
        )
        ttk.Button(model_frame, text="选择模型", command=self.select_model).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(model_frame, text="加载模型", command=self.load_model).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(model_frame, text="模型格式:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=(5, 0)
        )
        model_type_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_type,
            values=["pt", "onnx"],
            state="readonly",
            width=10,
        )
        model_type_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))

        # ========== 功能按钮区域 ==========
        func_frame = ttk.LabelFrame(main_frame, text="功能选择", padding="10")
        func_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(func_frame, text="图片检测", command=self.detect_image).grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Button(func_frame, text="摄像头检测", command=self.detect_camera).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(func_frame, text="视频检测", command=self.detect_video).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Button(func_frame, text="停止检测", command=self.stop_detection).grid(
            row=0, column=3, padx=5, pady=5
        )

        # 启用跟踪
        ttk.Checkbutton(
            func_frame,
            text="启用跟踪",
            variable=self.enable_tracking_var,
        ).grid(row=0, column=4, padx=5, pady=5)

        # 播放速度（只对视频生效）
        ttk.Label(func_frame, text="播放速度:").grid(
            row=0, column=5, padx=(15, 2), pady=5, sticky=tk.E
        )
        speed_combo = ttk.Combobox(
            func_frame,
            width=6,
            state="readonly",
            values=["0.5x", "1.0x", "1.5x", "2.0x"],
        )
        speed_combo.grid(row=0, column=6, padx=2, pady=5, sticky=tk.W)
        speed_combo.set("1.0x")

        def on_speed_change(event=None):
            text = speed_combo.get().replace("x", "")
            try:
                self.speed_var.set(float(text))
                logger.info("视频播放速度调整为 %sx", text)
            except ValueError:
                self.speed_var.set(1.0)
                logger.warning("播放速度解析失败，重置为 1.0x，原始值: %s", text)

        speed_combo.bind("<<ComboboxSelected>>", on_speed_change)

        # 保存检测视频选项
        ttk.Checkbutton(
            func_frame,
            text="保存检测视频",
            variable=self.save_video_var,
        ).grid(row=0, column=7, padx=(15, 2), pady=5)

        ttk.Button(
            func_frame,
            text="选择保存路径",
            command=self.select_save_path,
        ).grid(row=0, column=8, padx=5, pady=5)

        # ========== 显示区域 ==========
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(
            row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(display_frame, text="原图", padding="5")
        left_frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        self.original_label = ttk.Label(
            left_frame,
            text="请选择检测功能",
            background="black",
            foreground="white",
            anchor="center",
        )
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        right_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        self.result_label = ttk.Label(
            right_frame,
            text="检测结果将在此显示",
            background="black",
            foreground="white",
            anchor="center",
        )
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ========== 信息输出区域 ==========
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        self.info_text = tk.Text(info_frame, height=8)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(
            info_frame, orient="vertical", command=self.info_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)

    # ---------------- 定时轮询推理结果 ----------------

    def poll_results(self):
        try:
            result = self.controller.get_result()
            if result:
                original_img, annotated_img, det_result = result
                self.display_image(original_img, self.original_label)
                self.display_image(annotated_img, self.result_label)
                self.display_detection_info(det_result)

                # 如果开启了保存检测视频，则写入到 VideoWriter
                self.maybe_write_video(annotated_img)
        except Exception:
            logger.exception("[UI] 获取结果失败")
        finally:
            self.root.after(30, self.poll_results)

    # ---------------- 模型选择 / 加载 ----------------

    def select_model(self):
        current_type = self.model_type.get()
        extensions = {"pt": "*.pt", "onnx": "*.onnx"}
        file_types = [
            (f"{current_type.upper()} 模型", extensions.get(current_type, "*.*")),
            ("所有文件", "*.*"),
        ]
        file_path = filedialog.askopenfilename(
            title=f"选择 {current_type} 模型文件", filetypes=file_types
        )
        if file_path:
            logger.info("选择模型文件: %s", file_path)
            self.model_path.set(file_path)

    def load_model(self):
        try:
            path = self.model_path.get()
            if not path:
                messagebox.showerror("错误", "请先选择模型文件")
                return
            if not os.path.exists(path):
                messagebox.showerror("错误", "模型文件不存在")
                return

            logger.info("开始加载模型: %s (type=%s)", path, self.model_type.get())
            success, info_or_error = self.controller.load_model(
                path, self.model_type.get()
            )
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
                messagebox.showinfo(
                    "成功",
                    f"模型加载成功！\n名称: {model_name}\n任务: {task}\n设备: {device}",
                )
            else:
                logger.error("模型加载失败: %s", info_or_error)
                messagebox.showerror("错误", f"模型加载失败:\n{info_or_error}")
        except Exception:
            logger.exception("模型加载异常")
            messagebox.showerror("错误", "模型加载失败，请查看日志")

    # ---------------- 保存视频相关 ----------------

    def select_save_path(self):
        """选择检测视频保存路径（mp4 或 avi）"""
        file_path = filedialog.asksaveasfilename(
            title="选择保存视频路径",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 文件", "*.mp4"),
                ("AVI 文件", "*.avi"),
                ("所有文件", "*.*"),
            ],
        )
        if file_path:
            logger.info("选择输出视频路径: %s", file_path)
            self.save_path.set(file_path)

    def init_video_writer_if_needed(self, frame_shape):
        """
        创建 Windows 兼容性较好的 视频保存器（根据保存路径自动选择编码器）
        """
        if not self.save_video_var.get():
            return
        if not self.save_path.get():
            return
        if self.video_writer is not None:
            return

        h, w = frame_shape[:2]

        # 1. 自动选择编码器
        ext = os.path.splitext(self.save_path.get())[1].lower()

        if ext == ".avi":
            # Windows 通用编码器（兼容 99% 播放器）
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            codec_name = "MJPG"
        else:
            # 默认 MP4：使用 Windows 通用 H.264 (avc1) 编码器
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            codec_name = "avc1"

        # 2. FPS：视频 = 原 fps * speed，摄像头 = 摄像头 fps 或 25
        if self.current_fps is None or self.current_fps <= 0:
            fps = 25.0
        else:
            fps = (
                self.current_fps * max(self.speed_var.get(), 0.1)
                if self.is_video_mode
                else self.current_fps
            )

        logger.info(
            "初始化视频写入器: path=%s, codec=%s, fps=%.2f, size=(%d,%d)",
            self.save_path.get(),
            codec_name,
            fps,
            w,
            h,
        )

        # 3. 创建 VideoWriter
        self.video_writer = cv2.VideoWriter(
            self.save_path.get(), fourcc, fps, (w, h)
        )

        # 4. 写入失败保护
        if not self.video_writer.isOpened():
            logger.error("无法创建输出视频文件: %s", self.save_path.get())
            messagebox.showerror("错误", "无法创建输出视频，请更换保存路径")
            self.video_writer = None

    def maybe_write_video(self, annotated_img):
        """在有需要时将检测后的帧写入视频文件。"""
        if not self.save_video_var.get():
            return
        if not self.save_path.get():
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
            messagebox.showerror("错误", "请先加载模型")
            return

        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        try:
            logger.info("开始图片检测: %s", path)
            img_source = FrameSource(SourceType.IMAGE, path)
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
            messagebox.showerror("错误", f"图片检测失败:\n{str(e)}")

    # ---------------- 摄像头检测 ----------------

    def detect_camera(self):
        if self.controller.detector.adapter is None:
            messagebox.showerror("错误", "请先加载模型")
            return

        logger.info(
            "开始摄像头检测, 启用跟踪=%s",
            self.enable_tracking_var.get(),
        )

        # 设置跟踪开关
        self.controller.set_tracking(
            enabled=self.enable_tracking_var.get(),
            tracker_cfg="bytetrack.yaml",
        )

        # 如果需要保存但没有路径，提示选择
        if self.save_video_var.get() and not self.save_path.get():
            self.select_save_path()

        # 清理旧 writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.source = FrameSource(SourceType.CAMERA)
        if not self.source.open():
            logger.error("无法打开摄像头")
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_detecting = True
        self.is_video_mode = False  # 摄像头模式标记
        # 摄像头 fps
        try:
            self.current_fps = self.source.cap.get(cv2.CAP_PROP_FPS)
            if self.current_fps is None or self.current_fps <= 0:
                self.current_fps = 25.0
        except Exception:
            self.current_fps = 25.0

        logger.info("摄像头 FPS 估计为 %.2f", self.current_fps)

        self.controller.start_inference_thread()
        self.frame_generator = self.source.frames()
        self.camera_capture_loop()

    def camera_capture_loop(self):
        if not self.is_detecting:
            return

        try:
            flag, frame = next(self.frame_generator)
        except StopIteration:
            logger.info("摄像头帧源结束")
            self.is_detecting = False
            return

        if flag == "end":
            logger.info("摄像头帧源结束 (end flag)")
            self.is_detecting = False
            return

        if frame is not None:
            self.controller.submit_frame(frame)

        if self.is_detecting:
            # 摄像头就维持大约 30ms 的轮询频率即可
            self.root.after(30, self.camera_capture_loop)

    # ---------------- 视频检测 ----------------

    def detect_video(self):
        if self.controller.detector.adapter is None:
            messagebox.showerror("错误", "请先加载模型")
            return

        # 设置跟踪开关
        self.controller.set_tracking(
            enabled=self.enable_tracking_var.get(),
            tracker_cfg="bytetrack.yaml",
        )

        # 选择视频文件
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        logger.info(
            "开始视频检测: %s, 启用跟踪=%s",
            path,
            self.enable_tracking_var.get(),
        )

        # 如果需要保存但没有路径，提示选择
        if self.save_video_var.get() and not self.save_path.get():
            self.select_save_path()

        # 清理旧 writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.source = FrameSource(SourceType.VIDEO, path)
        if not self.source.open():
            logger.error("无法打开视频文件: %s", path)
            messagebox.showerror("错误", "无法打开视频文件")
            return

        self.is_detecting = True
        self.is_video_mode = True

        # 获取原视频 FPS
        try:
            fps = self.source.cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 25.0
            self.current_fps = fps
        except Exception:
            self.current_fps = 25.0

        logger.info("视频 FPS 读取为 %.2f", self.current_fps)

        self.controller.start_inference_thread()
        self.frame_generator = self.source.frames()
        self.video_capture_loop()

    def video_capture_loop(self):
        if not self.is_detecting:
            return

        try:
            flag, frame = next(self.frame_generator)
        except StopIteration:
            logger.info("视频帧源 StopIteration 结束")
            self.is_detecting = False
            return

        if flag == "end":
            logger.info("视频检测完成 (end flag)")
            self.is_detecting = False
            if self.source:
                self.source.release()
                self.source = None
            messagebox.showinfo("完成", "视频播放结束")
            return

        if frame is not None:
            self.controller.submit_frame(frame)

        if self.is_detecting:
            # 按原始 FPS 和播放速度控制读取帧的节奏
            fps = self.current_fps if self.current_fps and self.current_fps > 0 else 25.0
            speed = self.speed_var.get() if self.speed_var.get() > 0 else 1.0
            delay_ms = int(1000 / (fps * speed))
            if delay_ms < 1:
                delay_ms = 1
            self.root.after(delay_ms, self.video_capture_loop)

    # ---------------- 停止检测 ----------------

    def stop_detection(self):
        logger.info("收到停止检测请求")
        self.is_detecting = False
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

        self.original_label.configure(
            image="", text="请选择检测功能", background="black", foreground="white"
        )
        self.result_label.configure(
            image="", text="检测结果将在此显示", background="black", foreground="white"
        )
        self.info_text.delete(1.0, tk.END)

        self.original_tk_image = None
        self.result_tk_image = None

    # ---------------- 显示相关 ----------------

    def display_image(self, img, label):
        try:
            pil_img = Visualizer.resize_for_display(
                img, label.winfo_width(), label.winfo_height()
            )
            photo_image = ImageTk.PhotoImage(pil_img)
            label.configure(image=photo_image, text="", background="white")
            label.image = photo_image

            if label == self.original_label:
                self.original_tk_image = photo_image
            elif label == self.result_label:
                self.result_tk_image = photo_image
        except Exception:
            logger.exception("[Display] 显示图像失败")

    def display_detection_info(self, det_result):
        self.info_text.delete(1.0, tk.END)
        try:
            info_text = Visualizer.format_info_text(det_result)
            self.info_text.insert(tk.END, info_text)
        except Exception:
            logger.exception("[Display] 解析检测结果失败")
            self.info_text.insert(tk.END, "解析结果失败，详细信息请查看日志。\n")


if __name__ == "__main__":
    logger.info("应用启动")
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()
    logger.info("应用正常退出")
