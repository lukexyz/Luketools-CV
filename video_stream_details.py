"""
Extracting and printing video stream details
Example output:

Created capture: ../data/drop.avi
    Frame count: 182
    Frame width: 256
    Frame height: 240
    Frame rate: 30.0
Created capture: 0
    Frame count: -1
    Frame width: 640
    Frame height: 480
    Frame rate: 30.0
"""

import cv2

def print_capture_properties(*args):
    capture = cv2.VideoCapture(*args)
    print('Created capture:', ' '.join(map(str, args)))
    print('Frame count:', int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('Frame width:', int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame rate:', capture.get(cv2.CAP_PROP_FPS))
    
print_capture_properties('../data/drop.avi')
print_capture_properties(0)
