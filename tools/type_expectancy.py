from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    seek: int
    mode: str
    frame: np.ndarray | None
    fps_10:int
    using_sam:bool
