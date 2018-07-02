import cv2
import numpy as np

img = cv2.imread('data/circlesgrid.png', cv2.IMREAD_COLOR)
img = cv2.imread('data/book_angled_photo_small.png', cv2.IMREAD_COLOR)
show_img = np.copy(img)

selected_pts = []
drawing = False

# Added drawing tracers

def mouse_callback(event, x, y, flags, param):
    global selected_pts, show_img, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(show_img, (x, y), 10, (0, 255, 0), 3)
    
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        selected_pts.append([x, y])
        cv2.circle(show_img, (x, y), 10, (0, 255, 0), 3)
        
        
def select_points(image, points_num):
    global selected_pts
    selected_pts = []
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        cv2.imshow('image', image)

        k = cv2.waitKey(1)

        if k == 27 or len(selected_pts) == points_num:
            break

    cv2.destroyAllWindows()

    return np.array(selected_pts, dtype=np.float32)


show_img = np.copy(img)
src_pts = select_points(show_img, 4)
h, w = show_img.shape[0], show_img.shape[1]
dst_pts = np.array([[0, h], [0, 0], [h, 0], [h, h]], dtype=np.float32)

perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts)

unwarped_img = cv2.warpPerspective(img, perspective_m, (h, h))

cv2.imshow('result', np.hstack((show_img, unwarped_img)))
k = cv2.waitKey()

cv2.destroyAllWindows()
