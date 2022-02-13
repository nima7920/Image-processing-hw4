import q1_funcs
import matplotlib.pyplot as plt
import numpy as np
import cv2

input_path = "inputs/Points.txt"
output_paths = ["outputs/res01.jpg", "outputs/res02.jpg", "outputs/res03.jpg", "outputs/res04.jpg"]

''' showing initial points '''

n, points = q1_funcs.readPoints("inputs/Points.txt")
print(points)
for i in range(points[0].size):
    plt.plot(points[0, i], points[1, i], marker='.', color='blue')
plt.savefig(output_paths[0])
plt.show()

k = 2
colors = ["red", "green"]
for i in range(1, 3):
    selected_points = q1_funcs.get_random_centers(points, k)
    centers = q1_funcs.k_means(points, k)
    q1_funcs.draw_fig(points, centers, k, colors, output_paths[i])

''' clustering in polar coordinates '''
polar_points = q1_funcs.get_polar_coordinates(points)
polar_centers = q1_funcs.k_means_polar(polar_points, k)
q1_funcs.draw_polar_fig(polar_points, polar_centers, k, colors, output_paths[3])
