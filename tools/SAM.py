import cv2
import numpy as np
from ultralytics import FastSAM

from tools.type_expectancy import BornClicPoints, BoundingBox

def get_bbox_with_fastsam(predictor: FastSAM, frame, clic_points, use_zoom=False) -> BoundingBox|None :
    print(f'clic points = {clic_points}')

    # par défaut (pas de zoom) : pas de décalage
    x0, y0 = 0, 0
    point_coords = np.array(clic_points, dtype=np.int32)


    xmin = int(np.min(clic_points[:, 0]))
    xmax = int(np.max(clic_points[:, 0]))
    ymin = int(np.min(clic_points[:, 1]))
    ymax = int(np.max(clic_points[:, 1]))

    born_clic_points = BornClicPoints(xmin,xmax,ymin,ymax)
    if use_zoom:
        scale = 2

        # centre
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # demi-taille du patch (on force une taille min pour éviter un crop nul)
        dx = xmax - xmin
        dy = ymax - ymin
        half = int(max(dx, dy, 10.0) * scale / 2.0)

        h, w = frame.shape[:2]

        # bornes du patch dans l'image originale
        x0 = max(int(cx - half), 0)
        y0 = max(int(cy - half), 0)
        x1 = min(int(cx + half), w)
        y1 = min(int(cy + half), h)

        # crop
        frame = frame[y0:y1, x0:x1]

        # points convertis dans le repère du crop
        point_coords = np.array(
            [[int(p[0] - x0), int(p[1] - y0)] for p in clic_points],
            dtype=np.int32
        )

    labels = [1] * len(point_coords)

    results = predictor.predict(
        source=frame,
        points=point_coords.tolist(),
        labels=labels,
        retina_masks=True,  # masques alignés sur la résolution de l'image
        imgsz=max(frame.shape[0], frame.shape[1]),
        conf=0.4,
        iou=0.9
    )

    try:
        # masque dans le repère du frame (crop éventuel)
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        # bbox dans le repère du frame (crop)
        (x, y, bw, bh) = cv2.boundingRect(mask)
        bbox = BoundingBox((x,y,bw,bh))

        good_bbox = check_if_prediction_logic(born_clic_points, bbox)

        if not good_bbox:
            return None

        # conversion dans le repère de l'image originale
        bbox.x += x0
        bbox.y +=y0
        return bbox
    except Exception as e:
        print(f"FastSAM error: {e}")
        return None


def check_if_prediction_logic(born_clic_points: BornClicPoints, bbox: BoundingBox) -> bool:
    if abs(born_clic_points.x_max - born_clic_points.x_min) > bbox.bw:
        return False

    if bbox.bw > 3 * abs(born_clic_points.x_max - born_clic_points.x_min) :
        return False

    if abs(born_clic_points.y_max - born_clic_points.y_min) > bbox.bh:
        return False

    if bbox.bh > 3 * abs(born_clic_points.y_max - born_clic_points.y_min):
        return False

    return True
