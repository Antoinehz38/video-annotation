import cv2, os, time
from pathlib import Path
from functools import partial
from ultralytics import FastSAM

import logging

from tools.save_data import DataSaver
from tools.trackers import get_tracker
from tools.labelling_states.tracking.tracking_periode import tracking, tracker_init
from tools.labelling_states.normal_periode import normal_labelling
from tools.on_changes import on_change, on_change_fps, on_change_using_sam
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
    state = State(seek=0, mode="nav", frame=None, fps_10=5, using_sam=False,
                  predictor=FastSAM("./SAM_weight/FastSAM-s.pt"), tracker_mode=tracker_mode,
                  tracker=get_tracker(tracker_mode), last_box=None)

    data_saver = DataSaver(path_saving_folder=Path(saving_path))


    callback = partial(on_change, cap, state)

    # Setup trackbars
    cv2.createTrackbar("frame", "video", 0, max(total - 1, 0), callback)
    callback_fps = partial(on_change_fps, state)

    cv2.createTrackbar("tracking fps/10", "video", 0, 50, callback_fps)  # nouvelle trackbar
    cv2.setTrackbarPos("tracking fps/10", "video", 10)

    callback_using_sam = partial(on_change_using_sam, state)
    cv2.createTrackbar("using sam", "video", 0, 1, callback_using_sam)
    cv2.setTrackbarPos("using sam", "video", 1)
    cv2.displayStatusBar("video", f"")
    update_info(state)

    callback(0)

    while True:
        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')):  # quitter
            break
        if state.mode == "nav":
            if k == ord('t') and state.frame is not None:
                tracker_init(state)

            elif k == ord('n'):
                state.mode = 'normal'
                update_info(state)


        elif state.mode == "track":
            should_continue = tracking(cap, state, data_saver)
            time.sleep(10 / max(1, state.fps_10))
            if k == ord('b'):
                data_saver.erase_last()
                data_saver.erase_last()# because one press break when wrong prediction but another is done before breaking
                should_continue = False

            if not should_continue:
                tracker_init(state)

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
    main(video_path=video, tracker_mode='KCF')
