import cv2, os, time
import numpy as np
from pathlib import Path
from functools import partial
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import logging

from tools.save_data import DataSaver
from tools.trackers import get_tracker
from tools.labelling_states.tracking_periode import tracking
from tools.labelling_states.normal_periode import normal_labelling
from tools.on_changes import on_change, on_change_fps, on_change_using_sam
from tools.SAM import sam_colorize
from tools.type_expectancy import State
from tools.clean_video import clean_video
from tools.display_info import update_info

logger = logging.getLogger(__name__)


def main(video_path: Path, tracker_mode: str = 'MIL', saving_path: str = './dataset'):
    if not os.path.exists(video_path.with_name(video_path.stem + "cleaned.mp4")):
        clean_video(video_path)  # to remove black band on the edges

    cap = cv2.VideoCapture(str(video_path.with_name(video_path.stem + "cleaned.mp4")))
    if not cap.isOpened():
        logger.error("Erreur: vidéo introuvable")
        return

    # Setup
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    state = State(seek=0, mode="nav", frame=None, fps_10=5, using_sam=False)
    data_saver = DataSaver(path_saving_folder=Path(saving_path))
    tracker = get_tracker(tracker_type=tracker_mode)
    sam = sam_model_registry["vit_b"](checkpoint="./SAM_weight/sam_vit_b_01ec64.pth")

    mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.90, stability_score_thresh=0.96,
                                         crop_n_layers=0, min_mask_region_area=400, )
    callback = partial(on_change, cap, state)

    # Setup trackbars
    cv2.createTrackbar("frame", "video", 0, max(total - 1, 0), callback)
    callback_fps = partial(on_change_fps, state)

    cv2.createTrackbar("tracking fps/10", "video", 0, 50, callback_fps)  # nouvelle trackbar
    cv2.setTrackbarPos("tracking fps/10", "video", 5)

    callback_using_sam = partial(on_change_using_sam, state)
    cv2.createTrackbar("using sam", "video", 0, 1, callback_using_sam)
    cv2.displayStatusBar("video", f"")
    update_info(state)

    callback(0)

    while True:
        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')):  # quitter
            break
        if state.mode == "nav":
            if k == ord('t') and state.frame is not None:
                cv2.displayStatusBar("video",
                                     f"tracking mode draw a box for the target |enter to activate tracking",
                                     0)
                frame = state.frame.copy()
                if state.using_sam:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
                    masks = mask_gen.generate(img_rgb)
                    vis = sam_colorize(frame, masks, alpha=0.6, top_k=5)
                    cv2.imshow("video", vis)
                    bbox = cv2.selectROI("video", vis, fromCenter=False, showCrosshair=True)

                else:
                    bbox = cv2.selectROI("video", frame, fromCenter=False, showCrosshair=True)
                tracker = get_tracker(tracker_type=tracker_mode)  # previous tracker is no longer needed
                ok = tracker.init(frame, bbox)
                if ok:
                    if state.using_sam:
                        cv2.destroyWindow("SAM Segments")
                    state.mode = "track"
                    update_info(state)
                else:
                    logger.error('error occurred in the init of the tracker')

            elif k == ord('n'):
                state.mode = 'normal'
                update_info(state)


        elif state.mode == "track":

            should_continue = tracking(cap, tracker, state, data_saver)
            time.sleep(10 / max(1, state.fps_10))
            if not should_continue or k == ord('b'):
                data_saver.erase_last()
                data_saver.erase_last()  # because one press break when wrong prediction but another is done before breaking
                state.mode = "nav"
                update_info(state)
                cv2.setTrackbarPos("frame", "video", state.seek)
                callback(state.seek)

        elif state.mode == "normal":
            should_continue = normal_labelling(cap, state, data_saver)
            if not should_continue or k == ord('b'):
                state.mode = "nav"
                update_info(state)
                cv2.setTrackbarPos("frame", "video", state.seek)
                callback(state.seek)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = Path("./data/FPV_Quad_Wing_Chase_1.mp4")
    main(video_path=video, tracker_mode='MIL')
