import cv2
import logging

logger = logging.getLogger(__name__)


def get_tracker(tracker_type:str='KCF') -> cv2.legacy.Tracker|None:
    '''
    Set up tracker.
    Instead of KCF, you can also use 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
    '''

    if tracker_type == 'BOOSTING':
        return cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()
    if tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    if tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        return cv2.legacy.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    else:
        logger.error("tracker name undefined returning None")
        return None

