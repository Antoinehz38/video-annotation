from __future__ import annotations
import cv2
import numpy as np
import logging

from tools.type_expectancy import State, BoundingBox
from tools.save_data import DataSaver
from tools.trackers import get_tracker
from tools.display_info import update_info
from tools.labelling_states.tracking.tracking_tools import (get_area_to_sam, make_prediction,
                                                            make_click_from_last_box, waiting_for_validation)

logger = logging.getLogger(__name__)

def tracking(video, state:State, data_saver:DataSaver) -> bool:
    '''
    tracking handler
    :param video: cv2 cap of the wanted video
    :param state: state of the labeler
    :param data_saver: class to save the data to train a yolo after
    :return: a boolean if false the while loop should stop
    '''
    ok, frame = video.read()

    if not ok:
        return False
    state.frame = frame

    # Update tracker
    ok, bbox = state.tracker.update(frame)

    color = (255, 0, 0) if ok else (255, 255, 255)

    if not ok: # on tente de re init le tracking avec le yolo en faisant une prédiction de BBox
        print(f'lost mais on retente')
        new_clicks = make_click_from_last_box(state.last_box)
        new_bbox = make_prediction(state, frame, new_clicks)

        if new_bbox:
            state.tracker = get_tracker(state.tracker_mode)
            ok = state.tracker.init(frame, new_bbox.tuple)
            bbox = new_bbox.tuple


    if ok:
        # Tracking success
        data_saver.save_yolo_sample(frame, bbox)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2,color, 2, 1)
    else:

        return False

    cv2.putText(frame, " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.imshow("video", frame)
    state.seek = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('frame', 'video', state.seek)

    return True



def tracker_init(state:State) -> None :
    cv2.displayStatusBar("video",
                         f"tracking mode draw a box for the target |enter to activate tracking",
                         0)
    frame = state.frame.copy()
    if state.using_sam:
        click_points = get_area_to_sam(frame)
        if click_points is None:
            state.mode = 'nav'
            update_info(state)
            return None
        click_points = np.array(click_points)
        prediction = make_prediction(state, frame, click_points)
        if prediction:
            validation = waiting_for_validation()
        else:
            validation = False
        if not validation:
            cv2.displayStatusBar("video",
                                 f"No or wrong prediction | Draw yourself the bounding Box ",
                                 0)

            bbox = cv2.selectROI("video", frame, fromCenter=False, showCrosshair=True)
        else:
            bbox = prediction.tuple
    else:
        bbox = cv2.selectROI("video", frame, fromCenter=False, showCrosshair=True)
    state.tracker = get_tracker(tracker_type=state.tracker_mode)  # previous tracker is no longer needed
    ok = state.tracker.init(frame, bbox)
    if ok:
        state.last_box = BoundingBox(bbox)
        state.mode = "track"
        update_info(state)
    else:
        logger.error('error occurred in the init of the tracker')



