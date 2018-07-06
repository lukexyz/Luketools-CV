import cv2
import numpy as np

# ---------------------------------------- #
#         Load Images for Panorama         #
# ---------------------------------------- #

images = []
images.append(cv2.imread('data/panorama/0.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('data/panorama/1.jpg', cv2.IMREAD_COLOR))

print('Image shapes:')
for i, img in enumerate(images):
    print('\tImage', i, img.shape)


# ---------------------------------------- #
#         Create Panorama Sticher          #
# ---------------------------------------- #
"""
cv2.createStitcher builds an instance of the panorama stitching algorithm. To apply it
to the panorama creation, you need to call its stitch method. This method accepts an array
of images to combine, and returns a stitching result status as well as a panorama image. The
status may have one of the following values:
    cv2.STITCHER_OK
    cv2.STITCHER_ERR_NEED_MORE_IMGS
    cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL
    cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL
"""

frames = [cv2.resize(img, None, fx=0.25, fy=0.25) for img in images]

stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(frames)

if ret == cv2.STITCHER_OK:
    pano_preview = cv2.resize(pano, None, fx=0.5, fy=0.5)
    cv2.imshow('panorama', pano_preview)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('Error during stitching.')
    