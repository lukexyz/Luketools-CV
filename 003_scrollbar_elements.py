"""
Scrollbar Elements

Creates a window with three scroll bars, each corresponding to an RGB value.

    Slide the scroll bars to change the background colour. 
"""

import cv2
import numpy as np

cv2.namedWindow('window')

# Create a variable with the fill colour for the image
fill_val = np.array([255, 255, 255], np.uint8)

def trackbar_callback(idx, value):
    fill_val[idx] = value
    
# Create three trackbars and bind to a specific colour
cv2.createTrackbar('R', 'window', 255, 255, lambda v: trackbar_callback(2, v))
cv2.createTrackbar('G', 'window', 255, 255, lambda v: trackbar_callback(1, v))
cv2.createTrackbar('B', 'window', 255, 255, lambda v: trackbar_callback(0, v))
        
    
while True:
    image = np.full((500, 500, 3), fill_val)
    cv2.imshow('window', image)
    key = cv2.waitKey(3)
    
    # Press esc to quit window
    if key == 27:
        break

cv2.destroyAllWindows()
