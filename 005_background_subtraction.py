"""
Backgroud Subtraction

Removing the background from a scene with moving objects. 
Foreground and background are segmented and optional returns within the function call.

    Five segmentaion algorithms are explored:
    
    1. KadewTraKuPong and Bowden segmentaion mask
    2. Gaussian Mixture segmentation by Zoran Zuvkovic
    3. Godbeher, Matsukawa, and Goldberg
    4. Counting by Sagi Zeevi
    5. KNN Nearest Neighbors (best)
    
"""

import cv2
import numpy as np


def split_image_fgbg(subtractor, open_sz=(0,0), close_sz=(0,0),
                     show_bg=False, show_shdw=False):
    
    kernel_open = kernel_close = None
    
    if all(i > 0 for i in open_sz):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_sz)
    
    if all(i > 0 for i in close_sz):
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_sz)
        
    # Open video file
    cap = cv2.VideoCapture('data/traffic.mp4')

    while True:
        # --------------------- Get Video Frame and Status ------------------ #
        status_cap, frame = cap.read()
        if not status_cap:
            break
        
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        
        # ---------------------- Apply Segmentation Mask  ------------------- #
        
        fgmask = subtractor.apply(frame)
        objects_mask = (fgmask == 255).astype(np.uint8)
        shadows_mask = (fgmask == 127).astype(np.uint8)
        
        if kernel_open is not None:
            objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel_open)

        if kernel_close is not None:
            objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel_close)
            if kernel_open is not None:
                shadows_mask = cv2.morphologyEx(shadows_mask, cv2.MORPH_CLOSE, kernel_open)

        # ------------------- Split Foreground/Background ------------------- #
        
        foreground = frame
        foreground[objects_mask == 0] = 0
        
        if show_shdw:
            foreground[shadows_mask > 0] = (0, 255, 0)
        
        cv2.imshow('foreground', foreground)

        if show_bg:
            background = fgbg.getBackgroundImage()
            if background is not None:
                cv2.imshow('background', background)    
        
        # -------------------------- Flow Control --------------------------- #
        
        cv2.waitKey(10)                         # Wait 20ms to reduce framerate
        key = cv2.waitKey(3)
        if key == 27:
            print('Pressed Esc')
            break
            
    cap.release()
    cv2.destroyAllWindows()

"""
# 1. Create KadewTraKuPong and Bowden segmentaion mask
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# Call function and run background subtraction
split_image_fgbg(fgbg, (2, 2), (40, 40), show_bg=False, show_shdw=False)


# 2. Better version using Gaussian Mixture segmentation by Zoran Zuvkovic (1998)
fgbg = cv2.createBackgroundSubtractorMOG2()
split_image_fgbg(fgbg, (3, 3), (30, 30), show_bg=True, show_shdw=False)


# 3. Background subtraction algorithm of Godbeher, Matsukawa,
# and Goldberg to create background masks
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
split_image_fgbg(fgbg, (5, 5), (25, 25), show_bg=True)


# 4. "Counting" background subraction algorithm based (Sagi Zeevi)
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
split_image_fgbg(fgbg, (5, 5), (15, 15), show_bg=True)
"""

# 5. Nearest Neighbors background segmentation (best)
fgbg = cv2.createBackgroundSubtractorKNN()
split_image_fgbg(fgbg, (5, 5), (25, 25), True)
