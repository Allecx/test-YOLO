# core/source.py

import cv2
from enum import Enum
from typing import Generator, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SourceType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"


class FrameSource:
    """抽象帧源，根据源类型（图片、视频文件、摄像头）提供帧数据。"""

    def __init__(self, source_type: SourceType, path_or_id=None):
        """
        初始化 FrameSource。
        参数:
            source_type: SourceType，数据源类型 (IMAGE, VIDEO, CAMERA)
            path_or_id: 当类型为 VIDEO 时为视频文件路径，为 CAMERA 时可为摄像头设备ID（默认0），IMAGE时为图像路径。
        """
        self.source_type = source_type
        self.path_or_id = path_or_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False

    def open(self) -> bool:
        """
        打开视频或摄像头数据源。
        对于 IMAGE 类型无需打开。

        返回:
            bool: 如果成功打开（或IMAGE类型不需要打开）则返回 True，否则 False。
        """
        logger.info(
            "打开帧源: type=%s, path_or_id=%s",
            self.source_type,
            self.path_or_id,
        )

        if self.source_type == SourceType.CAMERA:
            # 打开默认摄像头（设备ID 0）
            self.cap = cv2.VideoCapture(0 if self.path_or_id is None else self.path_or_id)
        elif self.source_type == SourceType.VIDEO:
            # 打开视频文件
            self.cap = cv2.VideoCapture(self.path_or_id)
        else:
            # IMAGE 类型无需调用 open
            self.is_open = False
            return False

        self.is_open = self.cap.isOpened()
        if not self.is_open:
            logger.error(
                "打开帧源失败: type=%s, path_or_id=%s",
                self.source_type,
                self.path_or_id,
            )
        return self.is_open

    def frames(self) -> Generator[Optional[Tuple[Optional[str], Optional[object]]], None, None]:
        """
        帧生成器：逐帧产出 (flag, frame) 元组。

        对于 IMAGE 类型，仅产出一次 (None, image_frame)。
        对于 VIDEO/CAMERA 类型，循环读取视频/摄像头帧:
            每次 yield (None, frame)；当结束时 yield ('end', None) 以表示结束。

        返回:
            生成器，每次迭代返回 (flag, frame):
            - 正常帧: flag 为 None，frame 为图像帧 (numpy.ndarray)
            - 结束: flag 为 'end', frame 为 None
        """
        if self.source_type == SourceType.IMAGE:
            logger.debug("FrameSource.frames: 读取单张图像 %s", self.path_or_id)
            yield None, cv2.imread(self.path_or_id)
        else:
            logger.debug("FrameSource.frames: 开始连续读取帧 type=%s", self.source_type)
            while self.is_open:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("帧源读取结束或失败: type=%s", self.source_type)
                    break
                yield None, frame
            yield "end", None

    def release(self):
        """释放视频流/摄像头资源。"""
        if self.cap:
            logger.info(
                "FrameSource.release: 释放资源 type=%s, path_or_id=%s",
                self.source_type,
                self.path_or_id,
            )
            self.cap.release()
        self.is_open = False
