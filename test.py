import cv2
import numpy as np

input1 = cv2.imread("inputs/input.jpg")
input2 = cv2.imread("inputs/persoanl-photo1.jpg")

input2 = input2[:900, 100:-100, :].copy()
input2 = cv2.resize(input2, (240, 150))
input1[410:560,15:255,  :] = input2

cv2.imwrite("outputs/result.jpg", input1)
