import math
import random

import cv2
import numpy as np


def q4(logo):
    f = 500
    h, w = logo.shape[:2]

    rotX1 = np.pi
    rotY1 = 0
    rotZ1 = 0
    distX1 = 0
    distY1 = 0
    distZ1 = 25

    rotX2 = np.pi-0.89
    rotY2 = 0
    rotZ2 = 0
    distX2 = 0
    distY2 = -1200
    distZ2 = 25

    K = np.array([[f, 0, w / 2, 0],
                  [0, f, h / 2, 0],
                  [0, 0, 1, 0]])
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * f
    Kinv[-1, :] = [0, 0, 1]

    RX1 = np.array([[1, 0, 0, 0],
                    [0, np.cos(rotX1), -np.sin(rotX1), 0],
                    [0, np.sin(rotX1), np.cos(rotX1), 0],
                    [0, 0, 0, 1]])

    RY1 = np.array([[np.cos(rotY1), 0, np.sin(rotY1), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rotY1), 0, np.cos(rotY1), 0],
                    [0, 0, 0, 1]])

    RZ1 = np.array([[np.cos(rotZ1), -np.sin(rotZ1), 0, 0],
                    [np.sin(rotZ1), np.cos(rotZ1), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R1 = np.linalg.multi_dot([RX1, RY1, RZ1])
    T1 = np.array([[1, 0, 0, distX1],
                   [0, 1, 0, distY1],
                   [0, 0, 1, distZ1],
                   [0, 0, 0, 1]])

    RX2 = np.array([[1, 0, 0, 0],
                    [0, np.cos(rotX2), -np.sin(rotX2), 0],
                    [0, np.sin(rotX2), np.cos(rotX2), 0],
                    [0, 0, 0, 1]])

    RY2 = np.array([[np.cos(rotY2), 0, np.sin(rotY2), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rotY2), 0, np.cos(rotY2), 0],
                    [0, 0, 0, 1]])

    RZ2 = np.array([[np.cos(rotZ2), -np.sin(rotZ2), 0, 0],
                    [np.sin(rotZ2), np.cos(rotZ2), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R2 = np.linalg.multi_dot([RX2, RY2, RZ2])
    T2 = np.array([[1, 0, 0, distX2],
                   [0, 1, 0, distY2],
                   [0, 0, 1, distZ2],
                   [0, 0, 0, 1]])
    H1 = np.linalg.multi_dot([K, R1, T1, Kinv])
    H2 = np.linalg.multi_dot([K, R2, T2, Kinv])
    H = np.matmul(H2,np.linalg.inv(H1))
    H = np.linalg.inv(H)
    H = np.divide(H,H[2,2])

    shift = [
        [1, 0, 89],
        [0, 1, -178],
        [0, 0, 1]]
    H = np.matmul(shift,H)
    print('Result Matrix:')
    print(H)


    dst = cv2.warpPerspective(logo, H, (435,725))

    cv2.imwrite("res12.jpg", dst)


if __name__ == '__main__':
    logo = cv2.imread('logo.png')
    q4(logo)

    cv2.waitKey(0)
