import cv2
import numpy as np
from pathlib import Path


def clean_video(video_path:Path):
    outf = str(video_path.with_name(video_path.stem + "cleaned.mp4"))


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("Erreur ouverture vidéo.")

    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Frame illisible.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray.mean(axis=0) > 10
    x1, x2 = np.argmax(mask), len(mask) - np.argmax(mask[::-1]) - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = x2 - x1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outf, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[:, x1:x2])

    cap.release()
    out.release()
    print(f"Vidéo enregistrée: {outf}")
