import cv2
import random
import numpy as np

from tools.type_expectancy import State, BoundingBox
from tools.SAM import get_bbox_with_fastsam

def get_area_to_sam(frame:np.ndarray) -> list| None :
    cv2.displayStatusBar("video",
                         f"Clic multiple time on the target to activate the SAM| enter when finish clicking |'b' break",
                         0)
    click_points = []
    def on_mouse(event, x, y, flags, param):
        nonlocal click_points
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])

        if event == cv2.EVENT_RBUTTONDOWN:
            click_points.pop()
    cv2.setMouseCallback("video", on_mouse)

    while True:
        disp = frame.copy()
        for (x, y) in click_points:
            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("video", disp)
        kk = cv2.waitKey(10) & 0xFF

        if kk in (27, ord('b')):  # annuler
            return None

        if kk in (13, 10):
           break

    return click_points

def waiting_for_validation():
    cv2.displayStatusBar("video", f"enter to confirm the prediction | anything else to draw yourself ")
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (13, 10):      # Enter (13 = CR, 10 = LF selon plateformes)
            return True
        if k != 255:           # toute autre touche (255 = rien pressé)
            return False

def make_prediction(state:State, frame, click_points) -> BoundingBox|None :

    pred = get_bbox_with_fastsam(state.predictor, frame, click_points, use_zoom=False)
    if not pred:
        pred = get_bbox_with_fastsam(state.predictor, frame, click_points, use_zoom=True)

    if pred:
        x_full, y_full, bw, bh = pred.tuple
        cv2.rectangle(frame, (x_full, y_full), (x_full + bw, y_full + bh), (0, 255, 0), 2)
        cv2.imshow("video", frame)

    return  pred


def make_click_from_last_box(last_box: BoundingBox):
    x, y, w, h = last_box.tuple

    def sample_coord(start, length):
        center = start + length / 2
        if random.random() < 0.7:
            v = random.gauss(center, length * 0.15)
        else:
            v = random.uniform(start, start + length)
        v = int(max(start, min(v, start + length - 1)))
        return v

    points = []
    for _ in range(3):
        px = sample_coord(x, w)
        py = sample_coord(y, h)
        points.append([px, py])

    return  np.array(points, dtype=np.int32)
