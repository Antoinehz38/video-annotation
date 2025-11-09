import cv2

from tools.type_expectancy import State
from tools.save_data import DataSaver

drawing = False
ix, iy = -1, -1
bbox = None

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, bbox
    frame = param.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(frame, (ix, iy), (x, y), (255, 255, 0), 2)
        cv2.imshow("video", frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv2.rectangle(frame, (ix, iy), (x, y), (255, 255, 0), 2)
        cv2.imshow("video", frame)

def normal_labelling(video, state:State, data_saver:DataSaver):
    global bbox
    ret, frame = video.read()
    if not ret:
        return False

    bbox = None
    cv2.imshow("video", frame)
    cv2.setMouseCallback("video", draw_rect, frame)

    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == ord('b'):  # quitter sans valider
            bbox = None
            break
        if k == 13 and bbox is not None:  # entrée = valider
            break

    cv2.setMouseCallback("video", lambda *args: None)
    if bbox is None:
        return False

    x, y, w, h = map(int, bbox)
    data_saver.save_yolo_sample(frame, bbox)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2)
    cv2.imshow("video", frame)


    state.seek = int(video.get(cv2.CAP_PROP_POS_FRAMES)) + 1
    cv2.setTrackbarPos('frame', 'video', state.seek)
    return True
