import q3_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2

input_path = "inputs/slic.jpg"
output_paths = ["outputs/res06.jpg", "outputs/res07.jpg", "outputs/res08.jpg", "outputs/res09.jpg"]
img = cv2.imread(input_path)

shape = (img.shape[1], img.shape[0])
img_resized = cv2.resize(img, (1008, 776))
lab_vectors, xy_vectors = q3_funcs.get_feature_vectors(img_resized)
img_grads = q3_funcs.get_img_gradients(img_resized)

for i, n in enumerate([64, 256, 1024, 2048]):
    centers = q3_funcs.initialize_cluster_centers(img_resized, img_grads, n, 5)
    labels = q3_funcs.cluster_pixels(img_resized, centers, lab_vectors, xy_vectors, 0.5)
    labels = q3_funcs.remove_noise(labels, 3)

    labels = cv2.resize(labels, shape)
    result = q3_funcs.generate_result_img(img, labels)
    cv2.imwrite(output_paths[i], result)
