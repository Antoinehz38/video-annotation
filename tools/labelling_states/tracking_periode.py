import cv2

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

    # Start timer
    #timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
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

    #cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    cv2.imshow("video", frame)
    state.seek = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('frame', 'video', state.seek)

    return True
