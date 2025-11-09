import cv2
from tools.type_expectancy import State
from tools.display_info import update_info


def on_change(video, state:State, val):
    if state.mode != 'nav':
        return
    state.seek = int(val)
    video.set(cv2.CAP_PROP_POS_FRAMES, state.seek)
    ok, fr = video.read()
    if not ok:
        return
    state.frame = fr
    cv2.imshow("video", fr)

def on_change_fps(state:State, val:int):
    update_info(state)
    state.fps_10 = val

def on_change_using_sam(state:State, val:int):
    if val == 0:
        state.using_sam = False
        update_info(state)
    else:
        state.using_sam = True
        update_info(state)
