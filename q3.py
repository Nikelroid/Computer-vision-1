import math
import random

import cv2
import numpy as np


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def normalize(img):
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def q3(image1, image2):
    nullImage = np.zeros_like(image2)
    w1 = image1.shape[1]
    h1 = image1.shape[0]

    w2 = image2.shape[1]
    h2 = image2.shape[0]

    plate = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')

    plate[:h1, :w1] = image1
    plate[:h2, w1:w1 + w2] = image2

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    sift_image1 = cv2.drawKeypoints(image1, keypoints1, nullImage, (0, 255, 0))

    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    sift_image2 = cv2.drawKeypoints(image2, keypoints2, nullImage, (0, 255, 0))

    plate0 = plate.copy()

    plate0[:h1, :w1] = sift_image1
    plate0[:h2, w1:w1 + w2] = sift_image2

    cv2.imwrite('res13_corners.jpg', plate0)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:500]

    plate1 = plate0.copy()

    pointsFound = []

    for match in matches:
        p1 = np.array(keypoints1[match.queryIdx].pt, dtype='int')
        p2 = np.array(keypoints2[match.trainIdx].pt, dtype='int')
        pointsFound.append((p1, p2))
        cv2.circle(plate1, p1, 3, (255, 0, 0), thickness=2)
        cv2.circle(plate1, (p2[0] + w1, p2[1]), 3, (255, 0, 0), thickness=2)

    cv2.imwrite('res14_correspondences.jpg', plate1)

    sift_match = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches,
                                 nullImage, matchColor=(255, 0, 0), singlePointColor=(0, 255, 0))

    for match in matches:
        p1 = np.array(keypoints1[match.queryIdx].pt, dtype='int')
        p2 = np.array(keypoints2[match.trainIdx].pt, dtype='int')
        cv2.circle(sift_match, p1, 3, (255, 0, 0), thickness=2)
        cv2.circle(sift_match, (p2[0] + w1, p2[1]), 3, (255, 0, 0), thickness=2)

    cv2.imwrite('res15_match.jpg', sift_match)

    rand = random.choices(pointsFound, k=20)

    plate2 = plate.copy()

    for points in rand:
        cv2.circle(plate2, points[0], 3, (255, 0, 0), thickness=2)
        cv2.circle(plate2, (points[1][0] + w1, points[1][1]), 3, (255, 0, 0), thickness=2)
        cv2.line(plate2, points[0], (points[1][0] + w1, points[1][1]), (255, 0, 0), thickness=2)

    cv2.imwrite('res16.jpg', plate2)

    src_pts = np.int32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.int32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    print("RANSAC started")
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransacReprojThreshold=40, maxIters=30000,confidence=0.95)
    print("Homographic Matrix:")
    print(M)

    plate3 = plate.copy()

    for match in pointsFound:
        cv2.circle(plate3, match[0], 3, (255, 0, 0), thickness=1)
        cv2.circle(plate3, (match[1][0] + w1, match[1][1]), 3, (255, 0, 0), thickness=1)
        cv2.line(plate3, match[0], (match[1][0] + w1, match[1][1]), (255, 0, 0), thickness=1)

    for i in range(len(src_pts)):
        if np.sum(mask[i] != 0):
            cv2.circle(plate3, src_pts[i][0], 3, (0, 0, 255), thickness=2)
            cv2.circle(plate3, (dst_pts[i][0][0] + w1, dst_pts[i][0][1]), 3, (0, 0, 255), thickness=2)
            cv2.line(plate3, src_pts[i][0], (dst_pts[i][0][0] + w1, dst_pts[i][0][1]), (0, 0, 255), thickness=2)

    cv2.imwrite('res17.jpg', plate3)

    plate4 = plate.copy()
    cv2.imwrite('res18_mismatch.jpg', plate4)

    Mat = np.linalg.inv(M)

    h, w = image1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = np.array(cv2.perspectiveTransform(pts, Mat),dtype='int')
    img2 = image2.copy()

    for i in range(4):
        cv2.line(img2, dst[i - 1][0], dst[i][0], (255, 0, 0), 2)

    result = cv2.warpPerspective(image1, Mat, (img2.shape[1], img2.shape[0]))

    for i in range(len(result)):
        if np.max(result[i]) != 0:
            start_y = i
            break
    for i in range(len(result[0])):
        if np.max(result[:, i]) != 0:
            start_x = i
            break
    for i in range(len(result)):
        if np.max(result[result.shape[0] - i - 1]) != 0:
            end_y = result.shape[0] - i
            break
    for i in range(len(result[0])):
        if np.max(result[:, result.shape[1] - i - 1]) != 0:
            end_x = result.shape[1] - i
            break

    plate5 = np.zeros((img2.shape[0], img2.shape[1]+end_x-start_x, 3), dtype='uint8')
    plate5[:end_y-start_y, :end_x-start_x] = result[start_y:end_y, start_x:end_x]
    plate5[:, end_x-start_x:] = img2

    cv2.imwrite('res19.jpg', plate5)

    corners = [np.matmul(M,[0,0,1]),np.matmul(M,[image2.shape[0],0,1]),
               np.matmul(M,[image2.shape[0],image2.shape[1],1]),np.matmul(M,[0,image2.shape[1],1])]

    left = int(np.min(corners[:][0]))
    up = int(np.min(corners[:][1]))

    shift = [
        [1, 0, -up],
        [0, 1, -left],
        [0, 0, 1]]
    M = np.matmul(shift,M)

    result = cv2.warpPerspective(image2, M, (image2.shape[1] * 6, image2.shape[0] * 6))
    for i in range(len(result)):
        if np.max(result[i]) != 0:
            start_y = i
            break
    for i in range(len(result[0])):
        if np.max(result[:, i]) != 0:
            start_x = i
            break
    for i in range(len(result)):
        if np.max(result[result.shape[0] - i - 1]) != 0:
            end_y = result.shape[0] - i
            break
    for i in range(len(result[0])):
        if np.max(result[:, result.shape[1] - i - 1]) != 0:
            end_x = result.shape[1] - i
            break
    res20 = result[start_y:end_y, start_x:end_x]
    cv2.imwrite('res20.jpg', res20)

    img2 = cv2.warpPerspective(img2, M, (image2.shape[1] * 6, image2.shape[0] * 6))
    img2 = img2[start_y:end_y, start_x:end_x]
    plate5 = np.zeros((max(image1.shape[0], img2.shape[0]), image1.shape[1] + img2.shape[1], 3), dtype='uint8')
    plate5[:image1.shape[0], :image1.shape[1]] = image1
    plate5[:img2.shape[0], image1.shape[1]:] = img2
    cv2.imwrite('res21.jpg', plate5)


if __name__ == '__main__':
    image1 = cv2.imread('im03.jpg')
    image2 = cv2.imread('im04.jpg')

    q3(image1, image2)

    cv2.waitKey(0)
