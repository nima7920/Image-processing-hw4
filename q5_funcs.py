import cv2
import matplotlib.pyplot as plt
import numpy as np

import ffmpeg


def expand_contour(contour, ratio):
    result = []
    for i, point in enumerate(contour):
        point1 = contour[i]
        x1, y1 = point1[0], point1[1]
        # print("point 1:", x1, y1)
        point2 = contour[i - 1]
        x2, y2 = point2[0], point2[1]
        # print("point 2:", x2, y2)
        r, s = int((x1 - x2) / ratio), int((y1 - y2) / ratio)
        # print("r,s:", r, s)
        if r == 0 and s == 0:
            result.append(point1)
            continue
        for j in range(ratio):
            point3 = [x2 + j * r, y2 + j * s]
            # print("point 3:", point3)
            result.append(point3)
        result.append(point1)
        # print("#############")
    return result


def draw_contour(img, contour, color):
    new_contour = np.roll(contour, 1, axis=1)
    result = cv2.polylines(img.copy(), [new_contour], True, color, 1)
    return result


def get_image_grads(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # x_derivative = img_gray - np.roll(img_gray, 1, axis=0)
    # y_derivative = img_gray - np.roll(img_gray, 1, axis=1)
    # result = -np.square(x_derivative) + np.square(y_derivative)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
    result = (np.square(sobel_x) + np.square(sobel_y))
    return result


def calculate_external_energy(img_grads, contour):
    e_external = 0
    for point in contour:
        e_external -= img_grads[point[0], point[1]]
    return e_external


def calculate_average_distance(contour):
    shifted_contour = np.roll(contour, 1, axis=0)
    d = np.average(np.sum(np.square(contour - shifted_contour), axis=1))
    return d, shifted_contour


def calculate_curvature(contour, shifted_contour):
    shifted_contour2 = np.roll(shifted_contour, 1, axis=0)
    matrix = contour - 2 * shifted_contour + shifted_contour2
    d = np.sum(np.square(matrix))
    return d


def calculate_internal_energy(contour, alpha, beta):
    d, shifted_contour = calculate_average_distance(contour)
    curvature = calculate_curvature(contour, shifted_contour)
    e_internal = alpha * np.sum(np.square(np.sum(np.square(contour - shifted_contour), axis=1) - d)) + beta * curvature
    return e_internal


def calculate_total_energy(img_grads, contour, alpha, beta, landa):
    e_external = calculate_external_energy(img_grads, contour)
    e_internal = calculate_internal_energy(contour, alpha, beta)
    return e_internal + landa * e_external


def update_contour_greedy(img_grads, contour, alpha, beta, landa, search_radius):
    result, temp = contour.copy(), contour.copy()
    e = calculate_total_energy(img_grads, result, alpha, beta, landa)
    for i, point in enumerate(contour):
        temp = result.copy()
        for j in range(-search_radius, search_radius + 1):
            for k in range(-search_radius, search_radius + 1):
                temp[i] = [temp[i, 0] + j, temp[i, 1] + k]
                e1 = calculate_total_energy(img_grads, temp, alpha, beta, landa)
                if e1 < e:
                    result[i] = temp[i]
                    e = calculate_total_energy(img_grads, result, alpha, beta, landa)
    return result


def active_contours_greedy(img, init_contour, alpha, beta, landa):
    img_grads = get_image_grads(img)
    initial_energy = calculate_total_energy(img_grads, init_contour, alpha, beta, landa)
    contour = init_contour.copy()
    plt.imshow(draw_contour(img, contour, (0, 255, 0)))
    plt.show()
    for i in range(10):
        contour = update_contour_greedy(img_grads, contour, alpha, beta, landa, 1)
        plt.imshow(draw_contour(img, contour, (0, 255, 0)))
        plt.show()


''' #### methods for updating contour with dynamic programming #### '''


# def fill_neighbors(contour, search_radius):
#     m = (2 * search_radius + 1) ** 2
#     neighbors = np.zeros((contour.shape[0], m, contour.shape[1]))
#     n = 0
#     for i in range(-search_radius, search_radius + 1):
#         for j in range(-search_radius, search_radius + 1):
#             neighbors_ij = contour.copy()
#             neighbors_ij[:, 0] = neighbors_ij[:, 0].reshape((contour.shape[0], 1)) + i
#             neighbors_ij[:, 1] = neighbors_ij[:, 1].reshape((contour.shape[0], 1)) + j
#             neighbors[:, n, :] = neighbors_ij.reshape(contour.shape)
#             n += 1

def calculate_energy_between_points(point1, point2, img_grads, d, alpha, landa):
    e1 = -landa * img_grads[point2[0], point2[1]]
    e2 = alpha * np.square(np.sum(np.square(point1 - point2)) - d)
    return e1 + e2


def find_closest_neighbor(current_points, previous_point, energies, i, search_radius, img_grads, d, alpha, landa):
    index, energy = 0, float('inf')
    t = 0
    for j in range(-search_radius, search_radius + 1):
        for k in range(-search_radius, search_radius + 1):
            point = np.asarray([previous_point[0] + j, previous_point[1] + k])
            e = calculate_energy_between_points(current_points, point, img_grads, d, alpha, landa) + energies[
                i - 1, t]
            if e < energy:
                index = t
                energy = e
            t += 1
    return index, energy


def get_indices_energies_dp(img_grads, contour, alpha, landa, search_radius):
    m = (2 * search_radius + 1) ** 2
    n = float('inf')
    result_indices, result_energies = np.zeros((contour.shape[0], m)), np.zeros((contour.shape[0], m))
    d, shifted_contour = calculate_average_distance(contour)

    for r in range(m):  # iterating over first point's neighborhood
        temp_indices, temp_energies = np.zeros((contour.shape[0], m)), np.zeros((contour.shape[0], m))
        x, y = -1 + int(r / (2 * search_radius + 1)), -1 + (r % (2 * search_radius + 1))
        point = np.asarray([contour[0, 0] + x, contour[0, 1] + y])

        for i in range(m):
            temp_indices[1, i] = r
            point2 = np.asarray([contour[1, 0] + x, contour[1, 1] + y])
            temp_energies[1, i] = calculate_energy_between_points(point, point2, img_grads, d, alpha, landa)

        for i in range(2, contour.shape[0]):  # iterating over contour points
            t = 0
            for j in range(-search_radius, search_radius + 1):
                for k in range(-search_radius, search_radius + 1):
                    current_point = np.asarray([contour[i, 0] + j, contour[i, 1] + k])
                    previous_point = contour[i - 1].reshape(contour.shape[1], 1)
                    next_index, next_energy = find_closest_neighbor(current_point,
                                                                    previous_point,
                                                                    temp_energies, i, search_radius, img_grads, d,
                                                                    alpha,
                                                                    landa)
                    temp_indices[i, t] = next_index
                    temp_energies[i, t] = next_energy
                    t += 1
        previous_point = contour[-1].reshape(contour.shape[1], 1)
        index, energy = find_closest_neighbor(point, previous_point, temp_energies, contour.shape[0] - 1, search_radius,
                                              img_grads, d, alpha, landa)

        if energy < n:
            n = energy
            temp_energies[0, r] = energy
            temp_indices[0, r] = index
            result_energies = temp_energies.copy()
            result_indices = temp_indices.copy()

    return result_indices, result_energies


def update_contour_dp(contour, indices, search_radius):
    result = np.zeros(contour.shape, dtype='int')
    index = int(indices[1, 0])
    index_n = int(indices[0, index])
    x, y = -1 + int(index_n / (2 * search_radius + 1)), -1 + (index_n % (2 * search_radius + 1))
    result[-1, 0], result[-1, 1] = contour[-1, 0] + x, contour[-1, 1] + y
    for i in reversed(range(1, contour.shape[0])):
        index_n = int(indices[i, index_n])
        x, y = -1 + int(index_n / (2 * search_radius + 1)), -1 + (index_n % (2 * search_radius + 1))
        result[i - 1, 0], result[i - 1, 1] = int(contour[i - 1, 0] + x), int(contour[i - 1, 1] + y)
    return result


def active_contours_dp(img, init_contour, alpha, beta, landa, num_of_iter=10):
    img_grads = get_image_grads(img)
    results = []
    initial_energy = calculate_total_energy(img_grads, init_contour, alpha, beta, landa)
    contour = init_contour.copy()
    result = draw_contour(img, contour, (0, 0, 0))
    results.append(result)

    for i in range(num_of_iter):
        indices, energies = get_indices_energies_dp(img_grads, contour, alpha, landa, 1)
        contour = update_contour_dp(contour, indices, 1).astype('int')
        result = draw_contour(img, contour, (0, 0, 0))
        results.append(result)
        # plt.imshow(draw_contour(img, contour, (0, 255, 0)))
        # plt.show()

    return results


def generate_video(results, output_path, size, fps):
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for img in results:
        video.write(img)
    video.release()
