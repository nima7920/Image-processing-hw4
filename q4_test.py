import q4_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic, felzenszwalb

input_path = "inputs/birds.jpg"

img = cv2.imread(input_path)

q4_funcs.segment_SLIC(img)

# bird, mask = q4_funcs.get_sample_bird(img, 1910, 2200, 110, 75)
# match_x, match_y = q4_funcs.match_birds(img, bird, mask)

# segments = felzenszwalb(img, scale=800, sigma=0.6, min_size=100)
#
# plt.imshow(segments, cmap='gray')
# plt.show()
# plt.imsave("outputs/result-test.jpg", mark_boundaries(img, segments, color=(0, 0, 0)))
# plt.show()
