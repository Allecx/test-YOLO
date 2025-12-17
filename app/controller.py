# app/controller.py

"""
DetectionController：负责协调 UI 与核心推理逻辑。

- 持有 Detector 实例
- 管理推理线程与输入/输出队列
- 提供可选的“目标跟踪模式”
"""

import threading
import queue
import logging
from typing import Optional

from core.detector import Detector
from core.dto import DetectionResult


logger = logging.getLogger(__name__)


class DetectionController:
    """
    控制检测/跟踪流程：
    - 负责模型加载
    - 负责推理线程和输入/输出队列
    - 提供可选的“目标跟踪模式”
    """

    def __init__(self):
        self.detector = Detector()
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.thread = None
        self.stop_flag = False

        # 跟踪相关配置
        self.enable_tracking = False
        # 默认使用 ByteTrack 作为跟踪器（ultralytics 自带 tracker 配置）
        self.tracker_cfg = "bytetrack.yaml"
        self.imgsz = 640  # 统一推理尺寸

        logger.debug("DetectionController 实例化完成")

    # ---------- 公共接口 ----------

    def load_model(self, model_path: str, model_format: str):
        logger.info(
            "DetectionController: 请求加载模型 path=%s, format=%s",
            model_path,
            model_format,
        )
        success, info = self.detector.load_model(model_path, model_format)
        if success:
            logger.info(
                "模型加载成功: task=%s, device=%s",
                info.get("task"),
                info.get("device"),
            )
        else:
            logger.error("模型加载失败: %s", info)
        return success, info

    def set_tracking(self, enabled: bool, tracker_cfg: Optional[str] = None):
        """
        设置是否启用跟踪以及使用的 tracker 配置文件。

        参数:
            enabled: True 开启跟踪（会使用 model.track），False 则只做普通检测
            tracker_cfg: 如 "bytetrack.yaml", "botsort.yaml"，为空则沿用当前配置
        """
        self.enable_tracking = enabled
        if tracker_cfg is not None:
            self.tracker_cfg = tracker_cfg

        logger.info(
            "更新 tracking 配置: enabled=%s, tracker_cfg=%s",
            self.enable_tracking,
            self.tracker_cfg,
        )

    def start_inference_thread(self):
        """启动后台推理线程。"""
        if self.thread and self.thread.is_alive():
            logger.debug("推理线程已在运行，忽略重复启动请求")
            return
        self.stop_flag = False
        self.thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.thread.start()
        logger.info("推理线程启动")

    def stop_inference_thread(self):
        """
        停止后台推理线程，并清空队列。
        """
        if not self.thread:
            return

        logger.info("请求停止推理线程")
        self.stop_flag = True
        # 放一个 None 进去，确保线程能够尽快从队列中退出
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass

        self.thread.join(timeout=2)
        logger.info("推理线程已结束")

        # 清空队列
        with self.input_queue.mutex:
            self.input_queue.queue.clear()
        with self.output_queue.mutex:
            self.output_queue.queue.clear()

    def submit_frame(self, frame):
        """
        向推理线程提交一帧图像。

        为保证实时性，如果队列已满，会丢弃旧帧，只保留最新。
        """
        if self.detector.adapter is None:
            logger.warning("submit_frame 调用时模型尚未加载，忽略该帧")
            return

        try:
            # 如果已满，尝试丢弃旧数据
            if self.input_queue.full():
                try:
                    dropped = self.input_queue.get_nowait()
                    del dropped
                    logger.debug("输入队列已满，丢弃一帧旧数据")
                except queue.Empty:
                    pass

            self.input_queue.put_nowait(frame)
        except queue.Full:
            # 极端情况下仍可能满，直接丢弃最新帧
            logger.debug("输入队列仍然满，丢弃最新帧")

    def get_result(self):
        """
        非阻塞获取一帧推理结果。

        返回:
            (original_frame, annotated_frame, DetectionResult) 或 None
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    # ---------- 内部线程函数 ----------

    def _inference_worker(self):
        """
        推理线程主体：循环从 input_queue 中取帧，执行检测或跟踪，
        然后将结果放入 output_queue。
        """
        logger.info(
            "推理线程开始运行 (imgsz=%d, tracking=%s, tracker_cfg=%s)",
            self.imgsz,
            self.enable_tracking,
            self.tracker_cfg,
        )

        while not self.stop_flag:
            try:
                frame = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                logger.debug("推理线程收到结束标记 None，准备退出")
                break

            try:
                # 根据开关决定是纯检测还是跟踪模式
                if self.enable_tracking:
                    result = self.detector.track(
                        frame,
                        imgsz=self.imgsz,
                        tracker_cfg=self.tracker_cfg,
                        persist=True,
                    )
                else:
                    result = self.detector.infer(frame, imgsz=self.imgsz)

                annotated = result.plot()
                det_result = DetectionResult.from_yolo(result)

                # 输出队列同样保持最新一帧
                if self.output_queue.full():
                    try:
                        old = self.output_queue.get_nowait()
                        del old
                        logger.debug("输出队列已满，丢弃一帧旧结果")
                    except queue.Empty:
                        pass

                self.output_queue.put_nowait((frame, annotated, det_result))

            except Exception:
                logger.exception("推理线程处理帧时发生异常")
            finally:
                self.input_queue.task_done()

        logger.info("推理线程正常退出")
