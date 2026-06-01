"""
YOLOv8 wrapper.

Returns a list of `Detection`s per frame, restricted to the COCO
classes that are relevant to tourist-vs-local classification
(person, backpack, handbag, suitcase, cell phone).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    # xyxy in pixel coordinates of the frame
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


class Detector:
    """Thin wrapper around ultralytics.YOLO.

    Lazy-loads the model on first .infer() call so importing this
    module is cheap (useful for unit tests / dry runs)."""

    def __init__(self, model_path: str, conf_threshold: float,
                 class_map: Dict[str, int], device: str = "cpu"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.class_map = class_map                # name -> id
        self.id_to_name = {v: k for k, v in class_map.items()}
        self.allowed_ids = set(class_map.values())
        self.device = device
        self._model = None

    # ----- loading ---------------------------------------------------
    def load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is required.  pip install ultralytics"
            ) from exc
        log.info("Loading YOLO model: %s on %s",
                 self.model_path, self.device)
        self._model = YOLO(self.model_path)

    # ----- inference -------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        self.load()
        # `verbose=False` keeps the console clean
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        dets: List[Detection] = []
        if not results:
            return dets
        r = results[0]
        if r.boxes is None:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            if cls_id not in self.allowed_ids:
                continue
            dets.append(Detection(
                cls_id=int(cls_id),
                cls_name=self.id_to_name.get(int(cls_id), str(cls_id)),
                conf=float(conf),
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
            ))
        return dets
