import cv2

from tools.type_expectancy import State


def update_info(state:State):
    '''
    to display command available and the mode
    :param state:
    :return: None
    '''

    if state.mode == 'nav':
        cv2.displayStatusBar("video",
                             f"mode = {state.mode}, 't'-> tracking (SAM : {state.using_sam} | fps : {state.fps_10/10}) | 'n'-> normal labelling | arrow or slide to navigate",
                             0)
    elif state.mode == 'track':
        cv2.displayStatusBar("video",
                             f"mode = {state.mode}, fps = {state.fps_10/10}, 'b'-> break ",
                             0)
    elif state.mode == "normal":
        cv2.displayStatusBar("video",
                             f"mode = {state.mode}, 'b' -> break | draw box & enter to validate label",
                             0)
    else:
        cv2.displayStatusBar("video",
                             f"Unknown mode",
                             0)
