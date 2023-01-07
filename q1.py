import math

import cv2
import numpy as np


def normalize(img):
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def get_deretive_x(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]

    dx_blue = cv2.Sobel(blue, cv2.CV_16S, 1, 0)
    dx_green = cv2.Sobel(green, cv2.CV_16S, 1, 0)
    dx_red = cv2.Sobel(red, cv2.CV_16S, 1, 0)

    grad = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grad[i, j] = max(dx_blue[i, j], dx_green[i, j], dx_red[i, j])

    return grad


def get_deretive_y(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]

    dx_blue = cv2.Sobel(blue, cv2.CV_16S, 0, 1)
    dx_green = cv2.Sobel(green, cv2.CV_16S, 0, 1)
    dx_red = cv2.Sobel(red, cv2.CV_16S, 0, 1)

    grad = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grad[i, j] = max(dx_blue[i, j], dx_green[i, j], dx_red[i, j])

    return grad


def get_gradient(ix, iy):
    gradian = np.sqrt(ix + iy).astype('int')
    return gradian


def q1(image1, image2, sigma, size, k, s, n, astaneh, treshhold):
    gradx1 = get_deretive_x(image1)
    grady1 = get_deretive_y(image1)

    gradx2 = get_deretive_x(image2)
    grady2 = get_deretive_y(image2)

    i2x1 = gradx1 ** 2
    i2y1 = grady1 ** 2
    ixy1 = gradx1 * grady1

    i2x2 = gradx2 ** 2
    i2y2 = grady2 ** 2
    ixy2 = gradx2 * grady2

    gradian1 = get_gradient(i2x1, i2y1)
    gradian2 = get_gradient(i2x2, i2y2)

    cv2.imwrite('res01_grad.jpg', gradian1)
    cv2.imwrite('res02_grad.jpg', gradian2)

    kernel = cv2.getGaussianKernel(size, sigma)

    s2x1 = cv2.filter2D(i2x1, -1, kernel)
    s2y1 = cv2.filter2D(i2y1, -1, kernel)
    sxy1 = cv2.filter2D(ixy1, -1, kernel)

    s2x2 = cv2.filter2D(i2x2, -1, kernel)
    s2y2 = cv2.filter2D(i2y2, -1, kernel)
    sxy2 = cv2.filter2D(ixy2, -1, kernel)

    det1 = s2x1 * s2y1 - sxy1 ** 2
    trace1 = s2x1 + s2y1
    R1 = det1 - k * trace1

    det2 = s2x2 * s2y2 - sxy2 ** 2
    trace2 = s2x2 + s2y2
    R2 = det2 - k * trace2

    cv2.imwrite('res03_score.jpg', R1)
    cv2.imwrite('res04_score.jpg', R2)

    R1[R1 < treshhold] = 0
    R2[R2 < treshhold] = 0

    cv2.imwrite('res05_thresh.jpg', R1)
    cv2.imwrite('res06_thresh.jpg', R2)

    for y in range(R1.shape[0] - s):
        if np.max(R1[y:y + size, :]) == 0:
            y += s
            continue
        for x in range(R1.shape[1] - s):
            if np.max(R1[y:y + s, x:x + s]) == 0:
                continue
            t, val, t, loc = cv2.minMaxLoc(R1[y:y + s, x:x + s])
            R1[y:y + s, x:x + s] = np.zeros((s, s))
            R1[y + loc[1], x + loc[0]] = val
        print("R1 working: ", y, '/', R1.shape[0])

    for y in range(R2.shape[0] - s):
        if np.max(R2[y:y + size, :]) == 0:
            y += s
            continue
        for x in range(R2.shape[1] - s):
            if np.max(R2[y:y + s, x:x + s]) == 0:
                continue
            t, val, t, loc = cv2.minMaxLoc(R2[y:y + s, x:x + s])
            R2[y:y + s, x:x + s] = np.zeros((s, s))
            R2[y + loc[1], x + loc[0]] = val
        print("R2 working: ", y, '/', R2.shape[1])

    cv2.imwrite('res07_harris.jpg', R1)
    cv2.imwrite('res08_harris.jpg', R2)

    indexes1 = []
    indexes2 = []

    res1 = image1.copy()
    res2 = image2.copy()
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if R1[i, j] != 0:
                indexes1.append((j, i))
                cv2.circle(res1, (j, i), 5, (0, 255, 0), 3)
            if R2[i, j] != 0:
                indexes2.append((j, i))
                cv2.circle(res2, (j, i), 5, (0, 255, 0), 3)

    cv2.imwrite('test1.jpg', res1)
    cv2.imwrite('test2.jpg', res2)

    #______________________________________________________________________________

    r = int(n / 2)
    properties_vector_1 = []
    properties_vector_2 = []

    for i in range(R1.shape[0] - n):
        if max(R1[i + r, :]) == 0:
            continue
        for j in range(R1.shape[1] - n):
            if R1[i + r, j + r] > 0:
                properties_vector_1.append(image1[i:i + n, j:j + n].reshape(n**2 * 3))

    for i in range(R2.shape[0] - n):
        if max(R2[i + r, :]) == 0:
            continue
        for j in range(R2.shape[1] - n):
            if R2[i + r, j + r] > 0:
                properties_vector_2.append(image2[i:i + n, j:j + n].reshape(n**2 * 3))

    print(len(properties_vector_1))
    print(len(properties_vector_2))

    good_points_1 = []
    good_points_2 = []

    for point1 in range(len(properties_vector_1)):
        l = []
        for point2 in range(len(properties_vector_2)):
            l.append(np.linalg.norm(properties_vector_1[point1] - properties_vector_2[point2]))
        print("points1", point1, "/", len(properties_vector_1))
        d1_index = np.argmin(l)
        l = np.array(np.sort(l))
        d1 = l[0]
        d2 = l[1]

        try:
            if d1 / d2 < astaneh:
                good_points_1.append((point1, d1_index))
        except:
            print("d2=0")

    for point2 in range(len(properties_vector_2)):
        l = []
        for point1 in range(len(properties_vector_1)):
            l.append(np.linalg.norm(properties_vector_1[point1] - properties_vector_2[point2]))
        print("points2", point2, "/", len(properties_vector_2))
        d1_index = np.argmin(l)
        l = np.array(np.sort(l))
        d1 = l[0]
        d2 = l[1]
        try:
            if d1 / d2 < astaneh:
                good_points_2.append((point2, d1_index))
        except:
            print("d2=0")

    print(good_points_1)
    print(good_points_2)
    good_couples = []
    for gp1 in good_points_1:
        for gp2 in good_points_2:
            if gp1[0] == gp2[1] and gp1[1] == gp2[0]:
                good_couples.append(gp1)

    print('goods:', good_couples)
    checked = []
    bad_list = []
    for ck in good_couples:
        if ck[0] in checked:
            bad_list.append(ck[0])
        checked.append(ck[0])

    for bads in bad_list:
        for i in range(len(good_couples)):
            if bads == good_couples[i][0]:
                good_couples.pop(i)
                i -= 1

    checked = []
    bad_list = []
    for ck in good_couples:
        if ck[1] in checked:
            bad_list.append(ck[1])
        checked.append(ck[1])

    for bads in bad_list:
        for i in range(len(good_couples)):
            if bads == good_couples[i][1]:
                good_couples.pop(i)
                i -= 1

    res1 = image1.copy()
    res2 = image2.copy()
    for good_one in good_couples:
        cv2.circle(res1, (indexes1[good_one[0]]), 5, (0, 255, 0), 3)
        cv2.circle(res2, (indexes2[good_one[1]]), 5, (0, 255, 0), 3)
    cv2.imwrite('res09_corres.jpg', res1)
    cv2.imwrite('res10_corres.jpg', res2)

    w = res1.shape[1]
    h = res1.shape[0]

    plate = np.zeros((h + 100, 2 * w + 150, 3), dtype='uint8')

    plate[50:h + 50, 50:w + 50] = res1
    plate[50:h + 50, w + 100:2 * w + 100] = res2

    for good_one in good_couples:
        cv2.line(plate, (indexes1[good_one[0]][0] + 50, indexes1[good_one[0]][1] + 50),
                 (indexes2[good_one[1]][0] + w + 100, indexes2[good_one[1]][1] + 50), (255, 0, 0), 2)
    cv2.imwrite('res11.jpg', plate)


if __name__ == '__main__':
    image1 = cv2.imread('im01.jpg')
    image2 = cv2.imread('im02.jpg')

    q1(image1, image2, sigma=5, size=20, k=0.005, s=13, n=153, astaneh=0.971, treshhold=4*(10**7))

