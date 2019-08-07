import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_furthest(contour, center):
    dmax = 0
    index = 0
    for item in contour:
        # print(item)
        d = (item[0][0]-center[0])**2 + (item[0][1]-center[1])**2
        if d > dmax:
            dmax = d
            index = item
    return index[0]

# images = glob.glob('*.jpg')
# img = cv2.imread('acircles_pattern.png')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, points = cv2.findCirclesGrid(gray, (11, 4), flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
# # ret, corners = cv2.findChessboardCorners(gray, (4,3), None)
fig = plt.figure()
img0 = np.load('fingertip.npy')
plt.imshow(img0, cmap='jet')


# # gray = cv2.cvtColor(data0, cv2.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img0, (5, 5), 0)
ret, th = cv.threshold(blur, 30, 255, cv.THRESH_BINARY)
plt.imshow(blur)
plt.imshow(th)
contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
areas = [cv.contourArea(c) for c in contours]
# cv.drawContours(th, contours, -1, (0,255,0))
# plt.imshow(th)
# plt.show()

if areas:
    print(areas)
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv.boundingRect(cnt)
    # cv.rectangle(th, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center = [x+w/2, y+h/2]
    # print(x + w / 2)
    # print(y + h / 2)
    ftip = find_furthest(contours[max_index], center)
    print(ftip)
    th[ftip[1], ftip[0]] = 125
    plt.imshow(th)
    plt.show()
    # flag1 = findYLoc(y + h / 2)
    # flag2 = findXLoc(x + w / 2, flag1)
# area
# # skeleton
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv.erode(th, kernel, iterations=1)
# plt.imshow(erosion)
# plt.show()

# # hit-or-miss detection of fingertip
# kernel = np.array([[0, -1, 0], [0, 1, -1], [1, 1, 1]], np.uint8)
# img_output = np.array([[0] * 32 for _ in range(24)], np.uint8)
# erosion = cv.morphologyEx(th, cv.MORPH_HITMISS, kernel)
