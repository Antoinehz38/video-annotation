import cv2
import numpy as np
from segment_anything import SamPredictor




def sam_bbox_from_click(predictor:SamPredictor,frame, clic_points, scale=2):
    clic_points = np.asarray(clic_points, dtype=np.float32)

    print(f'clic points = {clic_points}')

    # étendue des points
    xmin = float(np.min(clic_points[:, 0]))
    xmax = float(np.max(clic_points[:, 0]))
    ymin = float(np.min(clic_points[:, 1]))
    ymax = float(np.max(clic_points[:, 1]))

    # centre
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # demi-taille du patch = écart max / 2 * scale
    dx = xmax - xmin
    dy = ymax - ymin
    half = int(max(dx, dy) * scale / 2)

    h, w = frame.shape[:2]

    # bornes du patch
    x0 = max(int(cx - half), 0)
    y0 = max(int(cy - half), 0)
    x1 = min(int(cx + half), w)
    y1 = min(int(cy + half), h)

    # crop
    crop_bgr = frame[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)

    predictor.set_image(crop_rgb)

    # point dans le repère du crop
    point_coords = np.array([[cx - x0, cy - y0]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    mask = masks[0].astype(np.uint8) * 255
    x, y, bw, bh = cv2.boundingRect(mask)

    # re-projeter dans l'image complète
    x_full = x0 + x
    y_full = y0 + y

    cv2.imshow('chill', crop_bgr)

    return (x_full, y_full, bw, bh)

