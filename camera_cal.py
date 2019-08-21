import cv2 as cv
import os
import imutils
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

path = './cal_data/'

cimg = list()
filenames = [filename for filename in os.listdir(path) if filename.endswith('0.jpg')]
for file in filenames:
    img_raw = cv.imread(path+file)
    img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(img_raw, 30, 255, cv.THRESH_BINARY)
    cimg.append(th)

objpoints = [[[0,0,0],[60,0,0],[120,0,0],
            [30,30,0],[90,30,0],
            [0,60,0],[60,60,0],[120,60,0],
            [30,90,0],[90,90,0]] for _ in cimg]
# plt.figure()
# plt.imshow(cimg[0])
# plt.show()
cal_points = list()
for i,img in enumerate(cimg):
    contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
    # cv.drawContours(img, contours, -1, (0, 255, 0), 5)
    # # contours = imutils.grab_contours(contours)
    img_points = list()
    contours = reversed(contours)
    for cnt in contours:
        M = cv.moments(cnt)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        img_points.append([cX, cY])
    cal_points.append(img_points)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(np.array(objpoints, np.float32)[:len(cal_points)], np.array(cal_points,np.float32), cimg[0].shape, None, None)
with open(path+'cal_matrix.pkl', 'wb') as file:
    pickle.dump([ret, mtx, dist, rvecs, tvecs], file)


timg = list()
filenames = [filename for filename in os.listdir(path) if filename.endswith('1.jpg')]
for file in filenames:
    img_raw = cv.imread(path+file)
    img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(img_raw, 30, 255, cv.THRESH_BINARY)
    timg.append(th)

for img in timg:
    plt.figure()
    plt.imshow(img)
    plt.title('origin')
    plt.show()

    h,  w = img.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # for img in cimg
    dst = cv.undistort(img, mtx, dist, None, None)
    plt.figure()
    plt.imshow(dst)
    plt.title('undistored')
    plt.show()








