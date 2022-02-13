import q5_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2

input_path = "inputs/tasbih.jpg"

img = cv2.imread(input_path)

# contour = np.asarray([[250, 250], [300, 220], [350, 210], [380, 250], [530, 290], [600, 330],
#                       [600, 420], [530, 520], [460, 620], [410, 690], [400, 800], [400, 1000]
#                          , [280, 970], [280, 600], [160, 520]], dtype='int32')
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
print(contour)
results = q5_funcs.active_contours_dp(img, contour, 0.001, 1, 1, 21)
q5_funcs.generate_video(results, "outputs/video.mp4", (img.shape[1], img.shape[0]), 3)
