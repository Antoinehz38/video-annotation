import cv2
import os
from pathlib import Path


class DataSaver:
    def __init__(self, path_saving_folder:Path=Path('./dataset')):
        self.path_saving_folder= path_saving_folder
        os.makedirs(path_saving_folder, exist_ok=True)
        self.img_id = len([f for f in os.listdir(path_saving_folder) if f.endswith(".jpg")])


    def save_yolo_sample(self, frame, bbox, class_id=0):
        """
        Sauvegarde une frame et sa bbox au format YOLO.
        path_saving_folder: dossier de sauvegarde
        frame: image (numpy array)
        bbox: (x, y, w, h) en pixels
        class_id: identifiant de la classe (int)
        """

        img_path = os.path.join(self.path_saving_folder, f"{self.img_id:06d}.jpg")
        label_path = os.path.join(self.path_saving_folder, f"{self.img_id:06d}.txt")

        H, W = frame.shape[:2]
        x, y, w, h = bbox
        xc = (x + w / 2) / W
        yc = (y + h / 2) / H
        ww = w / W
        hh = h / H

        cv2.imwrite(img_path, frame)
        with open(label_path, "w") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        self.img_id +=1

    def erase_last(self):

        self.img_id -=1
        img_path = os.path.join(self.path_saving_folder, f"{self.img_id:06d}.jpg")
        label_path = os.path.join(self.path_saving_folder, f"{self.img_id:06d}.txt")

        os.remove(img_path)
        os.remove(label_path)

