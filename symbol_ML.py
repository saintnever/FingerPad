import serial
import matplotlib

matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
from matplotlib import pyplot as plt
import queue
import threading
from matplotlib import animation
import seaborn as sns
import numpy as np
import time
# import MLX90640
import cv2 as cv
from collections import deque
import math
import colorsys
import os
import pickle
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import argparse
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import msvcrt
from collections import defaultdict
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
# def init_ORB(self):
#     self.orb = cv.ORB_create()
#     im = cv.imread('./heatlabel/archive/stop.jpg', cv.IMREAD_GRAYSCALE)
#     # im[im < 0.6*256] = 0
#     im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
#     im = cv.GaussianBlur(im, (31, 31), 0)
#     _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     self.kp1, self.des1 = self.orb.detectAndCompute(im, None)
#     self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#
# def HuM(self, im):
#     # contours, hierarchy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     # areas = [cv.contourArea(c) for c in contours]
#     # max_index = np.argmax(areas)
#     # cnt = contours[max_index]
#     # self.cnt = cnt
#     # (x, y, w, h) = cv.boundingRect(cnt)
#     # im = cv.resize(im[x:x+w][y:y+h], self.fsize1, interpolation=cv.INTER_CUBIC)
#     # Calculate Moments
#     moment = cv.moments(im)
#     # Calculate Hu Moments
#     huMoments = cv.HuMoments(moment)
#     re_huMoments = []
#     for i in range(len(huMoments)):
#         re_huMoments.append(-np.sign(huMoments[i][0]) * np.log10(abs(huMoments[i][0])))
#     re_huMoments.append(im.shape[0]/im.shape[1])
#     return np.array(re_huMoments)
#
# def init_sc(self):
#     self.sc = ShapeContext()
#     filenames = [filename for filename in os.listdir('./heatlabel/') if filename.endswith('.jpg') and not filename.startswith('img')]
#     kernel = np.ones((5, 5), np.uint8)
#
#     for n, filename in enumerate(filenames):
#         # plt.figure()
#         im = cv.imread('./heatlabel/'+filename, cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
#         # im = cv.bitwise_not(im)
#         im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
#         blur = cv.GaussianBlur(im, (31, 31), 0)
#         ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#         # th = cv.dilate(th, kernel, iterations=1)
#         # plt.imshow(th)
#         self.base.append(self.parse_img(th))
#         # points = self.sc.get_points_from_img(th, 15)
#         # descriptor = self.sc.compute(points).flatten()
#         # self.base.append(descriptor)
#         self.sym_names.append(filename.split('.')[0])
#     # plt.show()
#     # imt = cv.imread('./heatlabel/search.jpg', cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
#     # # imt = cv.bitwise_not(imt)
#     # imt = cv.resize(imt, self.fsize1, interpolation=cv.INTER_CUBIC)
#     # blurt = cv.GaussianBlur(imt, (31, 31), 0)
#     # ret, tht = cv.threshold(blurt, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#     # self.test = self.parse_img(tht)
#     # # tht = cv.dilate(tht, kernel, iterations=1)
#     # # ret, tht = cv.threshold(blurt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     # pointst = self.sc.get_points_from_img(tht, 15)
#     # descriptort = self.sc.compute(pointst).flatten()
#     # for i, base in enumerate(self.base):
#     #     # print(base.shape, descriptor.shape)
#     #     # res = cdist(np.array([base]), np.array([descriptort]), metric="cosine")
#     #     res = cdist(np.array([base]), np.array([descriptort]))
#     #     print(self.sym_names[i], res)
#     # print(np.array(self.base).shape, self.test.shape)
#     # print(self.match(np.array(self.base), self.test))
#
# def get_contour_bounding_rectangles(self, gray):
#     """
#       Getting all 2nd level bouding boxes based on contour detection algorithm.
#     """
#     contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     res = []
#     for cnt in contours:
#         (x, y, w, h) = cv.boundingRect(cnt)
#         res.append((x, y, x + w, y + h))
#     return res
#
# def get_contour_bounding_rectangles_largest(self, gray):
#     """
#       Getting all 2nd level bouding boxes based on contour detection algorithm.
#     """
#     contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     areas = [cv.contourArea(c) for c in contours]
#     max_index = np.argmax(areas)
#     cnt = contours[max_index]
#     self.cnt = cnt
#     (x, y, w, h) = cv.boundingRect(cnt)
#     # res = []
#     # for cnt in contours:
#     #     (x, y, w, h) = cv.boundingRect(cnt)
#     #     res.append((x, y, x + w, y + h))
#     return [(x, y, x + w, y + h)]
#
# def parse_img(self, img):
#     # invert image colors
#     # img = cv.bitwise_not(img)
#     # _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
#     # making numbers fat for better contour detectiion
#     # kernel = np.ones((2, 2), np.uint8)
#     # img = cv.dilate(img, kernel, iterations=1)
#
#     # getting our numbers one by one
#     rois = self.get_contour_bounding_rectangles_largest(img)
#     # print(len(rois))
#     grayd = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
#     nums = []
#     for r in rois:
#         grayd = cv.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
#         nums.append((r[0], r[1], r[2], r[3]))
#     # we are getting contours in different order so we need to sort them by x1
#     nums = sorted(nums, key=lambda x: x[0])
#     descs = []
#     for i, r in enumerate(nums):
#         if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
#             continue
#         im = cv.resize(img[r[1]:r[3], r[0]:r[2]], self.fsize1, interpolation=cv.INTER_CUBIC)
#         # blur = cv.GaussianBlur(im, (31, 31), 0)
#         # ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#         # points = self.sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 20)
#         points = self.sc.get_points_from_img(im, self.n_points)
#         descriptor = self.sc.compute(points).flatten()
#         descs.append(descriptor)
#     # print(descs)
#     return np.array(descs[0])
#
#
# def match(base, current):
#     """
#       Here we are using cosine diff instead of "by paper" diff, cause it's faster
#     """
#     res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
#     # res = cdist(base, current, metric="cosine")
#
#     idxmin = np.argmin(res.reshape(len(base)))
#     result = sym_names[idxmin]
#     return result, res.reshape(len(base))[idxmin]
#
# def init_ORB(self):
#     self.orb = cv.ORB_create()
#     im = cv.imread('./heatlabel/archive/stop.jpg', cv.IMREAD_GRAYSCALE)
#     # im[im < 0.6*256] = 0
#     im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
#     im = cv.GaussianBlur(im, (31, 31), 0)
#     _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     self.kp1, self.des1 = self.orb.detectAndCompute(im, None)
#     self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def HuM(im):
    # contours, hierarchy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # areas = [cv.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # self.cnt = cnt
    # (x, y, w, h) = cv.boundingRect(cnt)
    # im = cv.resize(im[x:x+w][y:y+h], self.fsize1, interpolation=cv.INTER_CUBIC)
    # Calculate Moments
    moment = cv.moments(im)
    # Calculate Hu Moments
    huMoments = cv.HuMoments(moment)
    re_huMoments = []
    for i in range(len(huMoments)):
        re_huMoments.append(-np.sign(huMoments[i][0]) * np.log10(abs(huMoments[i][0])))
    re_huMoments.append(im.shape[0]/im.shape[1])
    return np.array(re_huMoments)

def colorscale(self, datas, minc, maxc):
    data_scales = list()
    for data in datas:
        data_scale = int(256 * (data - minc) / (maxc - minc))
        # if data_scale < 0.5 * 256:
        if data_scale < 0 :
            data_scale = 0
        elif data_scale > 255:
            data_scale = 255
        data_scales.append(data_scale)
    # print(data, data_scale)
    return data_scales

if __name__ == '__main__':
    # user_set = ['ztx', 'xxh', 'hn', 'xxw']
    # user_set = ['zx_big', 'ztx_big']
    user_set = ['zx_XL', 'yh_XL', 'xxw_XL', 'wzy_XL', 'mth_XL', 'zjw_XL', 'ztx_XL', 'hn_XL']
    # symbol_set = ['qus', 'light']
    symbol_set = ['plus', 'minus', 'play', 'stop',  'search', 'qus']
    classname = ['Up', 'Down', 'Play', 'Stop',  'Search', 'Help']
    fsize1 = (32 * 10, 24 * 10)
    hu_size = (20 * 10, 20 * 10)
    HuM_symbol = defaultdict(list)
    symbol_ims = defaultdict(list)

    X_uset = []
    Y_uset = []
    timelist_s1 = []
    timelist_s2 = []
    for user in user_set:
        X = list()
        Y = list()
        path = './heatlabel/data/' + user + '/'
        filenames = [filename for filename in os.listdir(path) if filename.endswith('.jpg')]
        # ftimes1 = [ftime for ftime in os.listdir(path) if ftime.endswith('_s1.pkl')]
        # ftimes2 = [ftime for ftime in os.listdir(path) if ftime.endswith('_s2.pkl')]
        with open(path+user+'_s1.pkl', 'rb') as file:
            dts1 = pickle.load(file)
        with open(path + user + '_s2.pkl', 'rb') as file:
            dts2 = pickle.load(file)
        userlist1 = list()
        userlist2 = list()
        for i in range(0, len(dts1[0])):
            if dts1[0][i].split('/')[-1].split('_')[0] not in symbol_set or dts1[1][i] > 15:
                continue
            else:
                userlist1.append(dts1[1][i])

        for i in range(0, len(dts2[0])):
            if dts2[0][i].split('/')[-1].split('_')[0] not in symbol_set or dts2[1][i] > 15:
                continue
            else:
                userlist2.append(dts2[1][i])

        timelist_s1.append(userlist1)
        timelist_s2.append(userlist2)
        for filename in filenames:
            symbol = filename.split('_')[0]
            if symbol not in symbol_set:
                continue
            im = cv.imread(path + filename, cv.IMREAD_GRAYSCALE)
            # print(np.amax(im))
            # im[im < 0.5*255] = 0
            im = cv.resize(im, fsize1, interpolation=cv.INTER_CUBIC)
            im = cv.flip(im, 0)
            im = cv.GaussianBlur(im, (31, 31), 0)
            _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # _, im = cv.threshold(im, 255 * 0.6, 255, cv.THRESH_BINARY)
            kernel = np.ones((11, 11), np.uint8)
            im = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel)
            # im = cv.dilate(im, kernel)
            # im = cv.erode(im, kernel)
            contours, hierarchy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            areas = [cv.contourArea(c) for c in contours]
            if len(areas) == 0:
                continue
            max_index = np.argmax(areas)
            # print(areas[max_index], im.shape[0] * im.shape[1])

            cnt = contours[max_index]
            (x, y, w, h) = cv.boundingRect(cnt)
            # if (w * h) / (im.shape[0] * im.shape[1]) < 0.1:
            #     continue
            # th_crop = cv.resize(im[y:y + h, x:x + w], hu_size, interpolation=cv.INTER_CUBIC)
            symbol_ims[symbol].append(im[y:y + h, x:x + w])
            HuM_symbol[symbol].append(HuM(im[y:y + h, x:x + w]))
            X.append(HuM(im[y:y + h, x:x + w]))
            Y.append(symbol)
            # HuM_symbol[symbol].append(HuM(th_crop))
        X_uset.append(X)
        Y_uset.append(Y)

    # time duration
    tsum1, tsum2 = list(), list()
    for i in range(len(timelist_s1)):
        tsum1.append(np.mean(timelist_s1[i]))
        tsum2.append(np.mean(timelist_s2[i]))
        print(np.mean(timelist_s1[i]) >= np.mean(timelist_s2[i]),  np.mean(timelist_s1[i]),  np.mean(timelist_s2[i]))
    print('s1 time {}, s2 time {}'.format(np.mean(tsum1), np.mean(tsum2)))
    # create a ML model
    X_total = list()
    Y_total = list()

    for k, v in HuM_symbol.items():
        for hv in v:
            # print(hv)
            X_total.append(hv)
            Y_total.append(k)
    # svc = svm.SVC(gamma="scale")
    # parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
    cv_ml = StratifiedKFold(3, random_state=1, shuffle=True)
    # clf_svm = GridSearchCV(svc, parameters, cv=cv_ml)
    # print(cross_val_score(clf_svm, X_total, Y_total, cv=cv_ml))
    # # clf_svm.fit(X_total, Y_total)
    #
    # parameters = {'n_estimators': (1000, 2000, 3000), 'max_depth': [7, 9, 11], 'max_features':[3, 5, 7]}

    # clf_rf = GridSearchCV(RandomForestClassifier(random_state=0), parameters, cv=cv_ml)
    clf_rf = RandomForestClassifier(n_estimators= 1500, max_depth=9, random_state=0)
    scores_rf = cross_val_score(clf_rf, X_total, Y_total, cv=cv_ml)
    print("{0} RandomForest Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_rf, scores_rf.mean(), scores_rf.std() * 2))
    # clf_rf.fit(X_train, Y_train)
    X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.3, random_state=4)
    clf_rf.fit(X_train, Y_train)
    Y_predict = clf_rf.predict(X_test)
    # acc_between.append(np.sum(Y_predict == Y_test)/ len(Y_test))
    cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
    cfm_ratio = [list(item / np.sum(item)) for item in cfm]
    print(cfm_ratio)

    clf_gb = GradientBoostingClassifier()
    scores_gb = cross_val_score(clf_gb, X_total, Y_total, cv=cv_ml)
    print("{0} GradientBoosting Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_gb, scores_gb.mean(), scores_gb.std() * 2))

    # within user accuracy
    acc_within = list()
    cv = StratifiedKFold(2, random_state=1, shuffle=True)
    acc_matrix = np.zeros((len(symbol_set), len(symbol_set)))
    for i in range(len(X_uset)):
        X_total = X_uset[i]
        Y_total = Y_uset[i]
        # clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        clf_rf = RandomForestClassifier(n_estimators=1500, max_depth=9, random_state=0)
        acc_within.append(np.mean(cross_val_score(clf_rf, X_total, Y_total, cv=cv)))
        X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.3, random_state=4)
        clf_rf.fit(X_train, Y_train)
        Y_predict = clf_rf.predict(X_test)
        # acc_within.append(np.sum(Y_predict == Y_test) / len(Y_test))
        cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
        cfm_ratio = [list(item / np.sum(item)) for item in cfm]
        acc_matrix += cfm_ratio
        print('{} within accuracy is {}'.format(user_set[i], acc_within[-1]))

    print('Averaged within user accuracy is {}, std is {}'.format(np.mean(acc_within), np.std(acc_within)))
    plt.figure()
    df_cm = pd.DataFrame(acc_matrix / len(user_set), classname, classname)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm * 100, annot=True, annot_kws={'size': 14}, fmt='.1f', cmap="YlGnBu")
    plt.show()

    # between user accuracy
    acc_between = list()
    acc_matrix = np.zeros((len(symbol_set), len(symbol_set)))
    for i in range(len(X_uset)):
        X_test = X_uset[i]
        Y_test = Y_uset[i]
        X_train_total = [X_uset[m] for m in range(len(X_uset)) if m != i]
        Y_train_total = [Y_uset[m] for m in range(len(Y_uset)) if m != i]
        X_train = list()
        Y_train = list()
        for j in range(len(X_train_total)):
            for k in range(len(X_train_total[j])):
                X_train.append(X_train_total[j][k])
                Y_train.append(Y_train_total[j][k])
        # clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        clf_rf = RandomForestClassifier(n_estimators=2000, max_depth=11, random_state=0)
        clf_rf.fit(X_train, Y_train)
        Y_predict = clf_rf.predict(X_test)
        acc_between.append(np.sum(Y_predict == Y_test)/len(Y_test))
        cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
        cfm_ratio = [list(item / np.sum(item)) for item in cfm]
        acc_matrix += cfm_ratio
        # print(cfm_ratio)
        # acc_between.append(clf_rf.score(X_test, Y_test))
        print('{} between accuracy is {}'.format(user_set[i], acc_between[-1]))
    print('Averaged between user accuracy is {}, std is {}'.format(np.mean(acc_between), np.std(acc_between)))
    print('Averaged between user confusion matrix is {}'.format(acc_matrix / len(user_set)))
    plt.figure()
    df_cm = pd.DataFrame(acc_matrix / len(user_set), classname, classname)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm * 100, annot=True, annot_kws={'size': 14}, fmt='.1f', cmap="YlGnBu")
    plt.show()