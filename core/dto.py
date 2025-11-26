# core/dto.py

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Detection:
    """
    表示单个检测目标的数据结构。

    支持：
    - 普通检测：只用 class_id / class_name / confidence / bbox
    - 分割：额外 has_mask / mask_area
    - 跟踪：额外 track_id
    """

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        has_mask: bool = False,
        mask_area: Optional[float] = None,
        track_id: Optional[int] = None,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

        # 分割相关
        self.has_mask = has_mask
        self.mask_area = mask_area

        # 跟踪相关（仅在启用 tracking 时会被赋值）
        self.track_id = track_id


class DetectionResult:
    """封装检测结果（多个 Detection + names 映射）的类。"""

    def __init__(self, detections: List[Detection], names: dict = None):
        self.detections = detections
        self.names = names if names is not None else {}

    def is_empty(self) -> bool:
        return len(self.detections) == 0

    def count_by_class(self) -> dict:
        counts = {}
        for det in self.detections:
            name = det.class_name
            counts[name] = counts.get(name, 0) + 1
        return counts

    @classmethod
    def from_yolo(cls, result) -> "DetectionResult":
        """
        从 Ultralytics YOLO 的单帧结果对象构建 DetectionResult。

        支持：
        - 普通检测：使用 result.boxes
        - 分割：使用 result.boxes + result.masks
        - 跟踪：使用 result.boxes.id 获取 track_id
        """
        detections: List[Detection] = []
        names_map = result.names if hasattr(result, "names") else {}

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return cls(detections, names_map)

        # 分割掩码（如果有）
        masks = getattr(result, "masks", None)
        mask_data = None
        if masks is not None and hasattr(masks, "data"):
            try:
                mask_data = masks.data  # Tensor(N, H, W)
            except Exception:
                mask_data = None
                logger.debug("读取 masks.data 失败，忽略分割信息")

        # 跟踪 ID（只有启用了 model.track 才会有 boxes.id）
        track_id_list: List[Optional[int]] = []
        ids = getattr(boxes, "id", None)
        if ids is not None:
            try:
                track_id_list = ids.int().cpu().tolist()  # [id0, id1, ...]
            except Exception:
                # 兜底：保持长度一致，用 None 占位
                logger.debug("解析 boxes.id 失败，track_id 全部置为 None")
                track_id_list = [None] * len(boxes)
        else:
            track_id_list = [None] * len(boxes)

        # 优先使用 Ultralytics 的 Boxes 对象迭代
        try:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                cls_name = names_map.get(cls_id, str(cls_id))

                # 分割信息
                has_mask = False
                mask_area = None
                if mask_data is not None and idx < len(mask_data):
                    try:
                        m = mask_data[idx]  # Tensor(H, W)
                        m_np = m.cpu().numpy()
                        mask_area = float((m_np > 0.5).sum())
                        has_mask = True
                    except Exception:
                        has_mask = True
                        mask_area = None

                # 跟踪 ID（如果启用了 tracking）
                track_id = None
                if idx < len(track_id_list):
                    tid = track_id_list[idx]
                    track_id = int(tid) if tid is not None else None

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        has_mask=has_mask,
                        mask_area=mask_area,
                        track_id=track_id,
                    )
                )

            return cls(detections, names_map)

        except Exception:
            # 兜底：从 boxes.data / numpy 数组解析
            logger.exception("解析 YOLO Boxes 对象失败，退回到 ndarray 解析方式")

            data = getattr(boxes, "data", None)
            if data is None:
                try:
                    data = boxes.cpu().numpy()
                except Exception:
                    data = boxes

            import numpy as np

            arr = data if isinstance(data, np.ndarray) else np.array(data)
            for idx, det in enumerate(arr):
                if hasattr(det, "tolist"):
                    det = det.tolist()
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                conf = float(conf)
                cls_id = int(cls_id)
                cls_name = names_map.get(cls_id, str(cls_id))

                has_mask = False
                mask_area = None
                if mask_data is not None and idx < len(mask_data):
                    try:
                        m = mask_data[idx]
                        m_np = m.cpu().numpy()
                        mask_area = float((m_np > 0.5).sum())
                        has_mask = True
                    except Exception:
                        has_mask = True
                        mask_area = None

                track_id = None
                if idx < len(track_id_list):
                    tid = track_id_list[idx]
                    track_id = int(tid) if tid is not None else None

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        has_mask=has_mask,
                        mask_area=mask_area,
                        track_id=track_id,
                    )
                )

            return cls(detections, names_map)
