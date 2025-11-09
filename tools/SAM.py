import cv2
import numpy as np



def sam_colorize(image_bgr, masks, alpha=0.5, draw_contours=True, draw_labels=True, top_k=None):
    overlay = image_bgr.copy()

    if isinstance(masks, list) and len(masks) and isinstance(masks[0], dict):
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
    else:
        masks_sorted = masks
    if top_k:
        masks_sorted = masks_sorted[:top_k]

    rng = np.random.default_rng(42)
    colors = (rng.integers(0, 256, size=(len(masks_sorted), 3))).astype(np.uint8)

    for i, m in enumerate(masks_sorted):
        seg = m["segmentation"].astype(np.uint8) * 255  # HxW
        color = colors[i].tolist()

        color_img = np.zeros_like(image_bgr)
        color_img[:] = color
        overlay = cv2.addWeighted(overlay, 1.0, cv2.bitwise_and(color_img, color_img, mask=seg), alpha, 0)

        if draw_contours:
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (int(color[0]), int(color[1]), int(color[2])), 2)

        if draw_labels:
            x, y, w_, h_ = m["bbox"]
            label = f"id:{i} iou:{m.get('predicted_iou', 0):.2f}"
            cv2.rectangle(overlay, (x, y), (x + w_, y + h_), (int(color[0]), int(color[1]), int(color[2])), 2)
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - t_size[1] - 6), (x + t_size[0] + 6, y), (int(color[0]), int(color[1]), int(color[2])), -1)
            cv2.putText(overlay, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return overlay

