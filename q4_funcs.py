import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic, felzenszwalb

''' ### segmentation with SLIC #### '''


def segment_SLIC(img, segment_num=2048, compactness=2, max_iter=10, convert2lab=True):
    segments = slic(img, segment_num, compactness, max_iter, convert2lab)
    result = mark_boundaries(img, segments, (0, 0, 0))
    plt.imshow(result)
    plt.show()
    print(result)
    print(np.max(result), np.min(result))
    result = (result * 255).astype('uint8')

    cv2.imwrite("outputs/q4.jpg", result)


''' #### segmentation with Felzenswalb #### '''


def get_feature_vectors(img):
    # lab_vectors = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_vectors = img.copy()
    return lab_vectors


def segment_felzenszwalb(img, scale, sigma, min_size):
    labels = felzenszwalb(img, scale, sigma, min_size)
    result = mark_boundaries(img, labels, color=(0, 0, 0))
    return result, labels


def check_similarity(lab_vectors, super_pixel, sample, threshold):
    super_pixel_value = np.average(np.average(lab_vectors[super_pixel], axis=0), axis=0)
    diff = np.sqrt(np.sum(np.square(sample - super_pixel_value)))
    print(diff)
    if diff < threshold:
        return True
    return False


def find_matches(img, lab_vectors, labels, sample):
    result = img.copy()
    matches = []
    m, n = np.min(labels), np.max(labels)
    print(m, n)
    for i in range(m, n + 1):
        super_pixel_i = np.where(labels == i)
        if check_similarity(lab_vectors, super_pixel_i, sample, 60):
            matches.append(i)
            result[super_pixel_i] = 0
    return matches, result
