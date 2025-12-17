# core/visualizer.py
from PySide6.QtGui import QImage, QPixmap
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """
    负责图像缩放和检测结果文本格式化的工具类。

    - 不直接依赖 Tkinter，只返回 PIL.Image
    - 文本格式化支持 DetectionResult，兼容旧 dict 结构
    """

    @staticmethod
    def resize_for_display(cv_img, target_width: int, target_height: int):
        """将 OpenCV 图像缩放到适合显示的大小，保持宽高比。"""
        if cv_img is None:
            logger.error("resize_for_display 收到空图像")
            raise ValueError("提供的图像为空")

        h, w = cv_img.shape[:2]
        if w == 0 or h == 0:
            logger.error("resize_for_display 收到宽或高为 0 的图像: w=%d, h=%d", w, h)
            raise ValueError("图像宽或高为 0")

        # 防止 Label 初始宽高为 0
        if target_width <= 0 or target_height <= 0:
            logger.debug(
                "目标宽高为 0，使用原始尺寸: target_w=%d, target_h=%d",
                target_width,
                target_height,
            )
            target_width = w
            target_height = h

        ratio = min(
            target_width / w if target_width > 0 else 1.0,
            target_height / h if target_height > 0 else 1.0,
        )
        if ratio <= 0:
            ratio = 1.0

        new_w = max(int(w * ratio), 1)
        new_h = max(int(h * ratio), 1)

        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        return pil_img

    # @staticmethod
    # def resize_for_display(cv_img, target_width: int, target_height: int):
    #     h, w = cv_img.shape[:2]
    #     if h == 0 or w == 0:
    #         return QPixmap()
    #     ratio = min(target_width / w, target_height / h)
    #     new_w, new_h = int(w * ratio), int(h * ratio)
    #     resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    #     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #     qimg = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
    #     pixmap = QPixmap.fromImage(qimg.copy())  # 拷贝数据生成 QPixmap
    #     return pixmap

    @staticmethod
    def format_info_text(det_result) -> str:
        """
        根据检测结果生成文本说明。

        - 若 det_result 为 DetectionResult：
          - 输出：类别、置信度、坐标
          - 如有分割：输出“有分割掩码”、“掩码像素面积”
          - 如有跟踪：输出“跟踪ID”
        - 若为 dict（旧格式）：按 boxes + names 解析
        """
        lines = ["检测结果:"]
        class_count = {}

        # 新结构：DetectionResult
        if hasattr(det_result, "detections"):
            detections = det_result.detections
            if len(detections) == 0:
                lines.append("未检测到任何目标")
            else:
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    conf = det.confidence
                    cls_name = det.class_name

                    extra_parts = []

                    # 跟踪 ID（开启 tracking 时可用）
                    track_id = getattr(det, "track_id", None)
                    if track_id is not None:
                        extra_parts.append(f"跟踪ID: {track_id}")

                    # 分割掩码信息
                    if getattr(det, "has_mask", False):
                        extra_parts.append("有分割掩码")
                        if getattr(det, "mask_area", None) is not None:
                            extra_parts.append(f"掩码像素面积: {det.mask_area:.0f}")

                    extra_txt = ""
                    if extra_parts:
                        extra_txt = "，" + "，".join(extra_parts)

                    lines.append(
                        f"目标: {cls_name}, 置信度: {conf:.2f}, 位置: ({x1},{y1},{x2},{y2}){extra_txt}"
                    )
                    class_count[cls_name] = class_count.get(cls_name, 0) + 1

        # 旧结构：dict
        else:
            boxes = det_result.get("boxes", [])
            names = det_result.get("names", {})

            import numpy as np

            arr = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
            if arr is None or arr.size == 0:
                lines.append("未检测到任何目标")
            else:
                for det in arr:
                    if hasattr(det, "tolist"):
                        det = det.tolist()
                    x1, y1, x2, y2, conf, cls_id = det
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    conf = float(conf)
                    cls_id = int(cls_id)
                    cls_name = names.get(cls_id, str(cls_id))
                    lines.append(
                        f"目标: {cls_name}, 置信度: {conf:.2f}, 位置: ({x1},{y1},{x2},{y2})"
                    )
                    class_count[cls_name] = class_count.get(cls_name, 0) + 1

        if class_count:
            lines.append("")
            lines.append("统计信息:")
            for name, count in class_count.items():
                lines.append(f"{name}: {count} 个")

        return "\n".join(lines) + "\n"
