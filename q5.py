import q5_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2

input_path = "inputs/tasbih.jpg"

img = cv2.imread(input_path)

contour = []


def mouse_clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        contour.append([y, x])


cv2.imshow('img', img)
cv2.setMouseCallback('img', mouse_clicked)
cv2.waitKey()
contour = q5_funcs.expand_contour(contour, 20)
contour = np.asarray(contour, dtype='int32')
print(contour.shape)
results = q5_funcs.active_contours_dp(img, contour, 0.0001, 1, 1, 15)
q5_funcs.generate_video(results, "outputs/contour.mp4", (img.shape[1], img.shape[0]), 3)
cv2.imwrite("outputs/res11.jpg", results[-1])
