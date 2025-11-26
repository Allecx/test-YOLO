# infra/ultralytics_adapter.py

"""
UltralyticsAdapter：封装 Ultralytics YOLO 模型的底层调用逻辑。
"""

from typing import Optional
import logging

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class UltralyticsAdapter:
    """
    封装 Ultralytics YOLO 模型的适配器，用于加载模型并执行推理/跟踪。

    当前支持：
    - detect / segment / pose / obb 等任务的普通推理：model(...)
    - 多目标跟踪：model.track(..., persist=True, tracker=...)
      persist=True 时，Ultralytics 会在连续帧间保持轨迹状态
    """

    def __init__(self):
        self.model: Optional[YOLO] = None  # Ultralytics YOLO 模型实例

    def load_model(self, model_path: str):
        """
        加载 Ultralytics YOLO 模型。

        参数:
            model_path: 模型文件路径 (.pt 或 .onnx)

        返回:
            (success: bool, info_or_error):
              - True 和模型信息字典{'task': ..., 'device': ..., 'names': ...} 如果加载成功
              - False 和错误信息字符串 如果加载失败
        """
        try:
            logger.info("UltralyticsAdapter: 开始加载模型 %s", model_path)

            # 使用 Ultralytics 提供的 YOLO 类加载模型（自动处理 .pt / .onnx）
            self.model = YOLO(model_path)

            # 选择设备：如果有 GPU 则用 CUDA，否则 CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model.to(device)
            except Exception:
                # 某些 ONNX 模型在 .to('cuda') 时会报错，回退到 CPU
                logger.warning("模型迁移到 %s 失败，回退到 CPU", device)
                device = "cpu"

            task = getattr(self.model, "task", None)
            names = getattr(self.model, "names", None)
            info = {"task": task, "device": device, "names": names}

            logger.info(
                "Ultralytics 模型加载成功: task=%s, device=%s, classes=%d",
                task,
                device,
                len(names) if isinstance(names, (list, dict)) else -1,
            )
            return True, info
        except Exception as e:
            logger.exception("UltralyticsAdapter.load_model 失败")
            return False, str(e)

    def infer(self, image, imgsz=640):
        """
        对单帧图像执行普通推理（无跟踪）。

        返回:
            Ultralytics Results 对象（单帧）
        """
        if self.model is None:
            logger.error("infer 在模型未加载时被调用")
            raise RuntimeError("模型未加载")

        logger.debug("UltralyticsAdapter: infer 调用 imgsz=%d", imgsz)
        results = self.model(image, imgsz=imgsz)
        return results[0] if isinstance(results, (list, tuple)) else results

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
        对单帧图像执行多目标跟踪推理。

        说明：
        - 使用 Ultralytics 的 model.track 接口，并设置 persist=True，
          这样在多次调用之间会维持轨迹状态和 track_id

        参数:
            image: 单帧图像 (numpy.ndarray, BGR)
            imgsz: 推理尺寸
            tracker_cfg: 跟踪器配置文件，如 "bytetrack.yaml" 或 "botsort.yaml"
            persist: 是否在多帧之间保持 tracks（通常保持 True）
            conf: 置信度阈值（可选）
            iou: IoU 阈值（可选）

        返回:
            Ultralytics Results 对象（单帧，包含 boxes.id 等跟踪信息）
        """
        if self.model is None:
            logger.error("track 在模型未加载时被调用")
            raise RuntimeError("模型未加载")

        kwargs = {
            "imgsz": imgsz,
            "persist": persist,
        }
        if tracker_cfg is not None:
            kwargs["tracker"] = tracker_cfg
        if conf is not None:
            kwargs["conf"] = conf
        if iou is not None:
            kwargs["iou"] = iou

        logger.debug(
            "UltralyticsAdapter: track 调用 imgsz=%d, tracker=%s, persist=%s, conf=%s, iou=%s",
            imgsz,
            tracker_cfg,
            persist,
            conf,
            iou,
        )

        results = self.model.track(image, **kwargs)
        return results[0] if isinstance(results, (list, tuple)) else results
