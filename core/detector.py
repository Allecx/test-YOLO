# core/detector.py

"""
Detector：对上层提供统一的模型加载 / 推理 / 跟踪接口，
内部通过 UltralyticsAdapter 与 Ultralytics YOLO 交互。
"""

from typing import Optional
import logging

from infra.ultralytics_adapter import UltralyticsAdapter

logger = logging.getLogger(__name__)


class Detector:
    """
    通用检测器类，封装模型加载 / 推理 / 跟踪逻辑。

    - 内部通过 UltralyticsAdapter 与 Ultralytics YOLO 交互
    - 对上提供统一的 infer() 和 track() 接口
    """

    def __init__(self):
        self.adapter: Optional[UltralyticsAdapter] = None
        self.task: Optional[str] = None  # 模型任务类型，例如 'detect', 'segment', 'pose', 'obb'
        self.names = None  # 类别名称字典
        logger.debug("Detector 初始化完成")

    def load_model(self, model_path: str, model_format: str):
        """
        根据给定路径和格式加载模型。

        当前支持：
        - .pt
        - .onnx
        返回: (success: bool, info_or_error)
        """
        logger.info("Detector.load_model: path=%s, format=%s", model_path, model_format)

        if model_format == "pth":
            msg = "不支持 .pth 模型格式，请使用 .pt 或 .onnx"
            logger.warning("尝试加载不支持的 pth 模型: %s", model_path)
            return False, msg

        try:
            adapter = UltralyticsAdapter()
            success, info_or_error = adapter.load_model(model_path)
            if success:
                self.adapter = adapter
                self.task = info_or_error.get("task")
                self.names = info_or_error.get("names")
                logger.info(
                    "模型加载成功: task=%s, device=%s",
                    self.task,
                    info_or_error.get("device"),
                )
                return True, info_or_error
            else:
                logger.error("UltralyticsAdapter 加载模型失败: %s", info_or_error)
                return False, info_or_error
        except Exception:
            logger.exception("Detector.load_model 过程中发生异常")
            return False, "加载模型时发生异常，请查看日志"

    def infer(self, image, imgsz: int = 640):
        """
        执行单帧普通推理（无跟踪）。

        返回:
            Ultralytics Results 对象
        """
        if self.adapter is None:
            logger.error("infer 在模型未加载时被调用")
            raise RuntimeError("模型未加载")

        logger.debug("执行普通推理 imgsz=%d", imgsz)
        return self.adapter.infer(image, imgsz=imgsz)

    def track(
        self,
        image,
        imgsz: int = 640,
        tracker_cfg: Optional[str] = None,
        persist: bool = True,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        """
        执行单帧多目标跟踪推理。

        通常在摄像头 / 视频流中连续多次调用，并设置 persist=True，
        以便 Ultralytics 在内部对轨迹进行持续维护。

        返回:
            Ultralytics Results 对象（包含 boxes.id 等跟踪信息）
        """
        if self.adapter is None:
            logger.error("track 在模型未加载时被调用")
            raise RuntimeError("模型未加载")

        logger.debug(
            "执行跟踪推理 imgsz=%d, tracker_cfg=%s, persist=%s, conf=%s, iou=%s",
            imgsz,
            tracker_cfg,
            persist,
            conf,
            iou,
        )
        return self.adapter.track(
            image,
            imgsz=imgsz,
            tracker_cfg=tracker_cfg,
            persist=persist,
            conf=conf,
            iou=iou,
        )
