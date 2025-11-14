import cv2, time
import numpy as np

from tools.type_expectancy import State
from tools.save_data import DataSaver


def tracking(video, tracker:cv2.legacy.Tracker, state:State, data_saver:DataSaver) -> bool:
    '''
    tracking handler
    :param video: cv2 cap of the wanted video
    :param tracker: tracker chosen
    :param state: state of the labeler
    :param data_saver: class to save the data to train a yolo after
    :return: a boolean if false the while loop should stop
    '''
    ok, frame = video.read()
    if not ok:
        return False


    # Update tracker
    ok, bbox = tracker.update(frame)

    if ok:
        # Tracking success
        data_saver.save_yolo_sample(frame, bbox)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return False

    cv2.putText(frame, " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.imshow("video", frame)
    state.seek = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('frame', 'video', state.seek)

    return True


def get_area_to_sam(frame:np.ndarray):
    cv2.displayStatusBar("video",
                         f"Clic multiple time  on the target to activate the SAM | enter when finish clicking",
                         0)
    click_points = []
    time_start = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click_points, time_start
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            if not time_start:
                time_start = time.time()


    cv2.setMouseCallback("video", on_mouse)

    while True:
        disp = frame.copy()
        for (x, y) in click_points:
            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("video", disp)
        kk = cv2.waitKey(10) & 0xFF

        if kk in (27, ord('q')):  # annuler
            break

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
