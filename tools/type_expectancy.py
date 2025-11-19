from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import FastSAM



@dataclass
class State:
    seek: int
    mode: str
    frame: np.ndarray | None
    fps_10:int
    using_sam:bool
    predictor: FastSAM
    tracker_mode: str
    tracker: cv2.legacy.Tracker
    last_box: BoundingBox |None


@dataclass
class BornClicPoints:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class BoundingBox:
    def __init__(self, *args):
        if len(args) == 1:
            self.x, self.y, self.bw, self.bh = args[0]
        elif len(args) == 4:
            self.x, self.y, self.bw, self.bh = args
        else:
            raise ValueError("Invalid args")

    @property
    def tuple(self):
        return self.x, self.y, self.bw, self.bh
