import q4_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic, felzenszwalb

input_path = "inputs/birds.jpg"
output_path = "outputs/res10.jpg"
img = cv2.imread(input_path)

result, labels = q4_funcs.segment_felzenszwalb(img, 500, 0.9, 100)

lab_vectors = q4_funcs.get_feature_vectors(img)

# sample_img = img[1910:2020, 2200:2275, :]
# result1, labels1 = q4_funcs.segment_felzenszwalb(sample_img, 800, 0.6, 100)
# plt.imshow(result1)
# plt.show()

sample = np.asarray([105, 105, 105])
# sample = q4_funcs.calculate_sample_value(lab_vectors, 1910, 2200, 110, 75)
matches, result = q4_funcs.find_matches(img[1500:2350, 220:, :], lab_vectors[1500:2350, 220:, :],
                                        labels[1500:2350, 220:],
                                        sample)
# q4_funcs.draw_rectangles(result, labels, matches, (110, 75), (1500, 220))
plt.imshow(result)
plt.show()
img_copy = img[1500:2350, 220:, :].copy()
mask = np.where(result == 0)
img_copy[result != 0] = 0

plt.imshow(img_copy)
plt.show()
final_result = np.zeros(img.shape, dtype='uint8')
final_result[1500:2350, 220:, :] = img_copy
cv2.imwrite(output_path, final_result)
