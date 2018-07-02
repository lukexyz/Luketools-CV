import cv2
import numpy as np

# Detail openCV version and load image
print(cv2.__version__)

image_bgr = cv2.imread('data/Lena.png', cv2.IMREAD_COLOR)
print("\nLena.png shape:", image_bgr.shape)

# Transfer image into four-dimensional floating point tensor
image_bgr_float = image_bgr.astype(np.float32)
image_rgb = image_bgr_float[..., ::-1]
tensor_chw = np.transpose(image_rgb, (2, 0, 1))
tensor_nchw = tensor_chw[np.newaxis, ...]

# Output tensor shape
print("\n  Tensor shape:", tensor_nchw.shape)
