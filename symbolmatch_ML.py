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
from shape_context import ShapeContext

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
def get_contour_bounding_rectangles(self, gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        res.append((x, y, x + w, y + h))
    return res
#
def get_contour_bounding_rectangles_largest(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    # self.cnt = cnt
    (x, y, w, h) = cv.boundingRect(cnt)
    # res = []
    # for cnt in contours:
    #     (x, y, w, h) = cv.boundingRect(cnt)
    #     res.append((x, y, x + w, y + h))
    return [(x, y, x + w, y + h)]

def parse_img(img, sc, n_points):
    # invert image colors
    # img = cv.bitwise_not(img)
    # _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    # making numbers fat for better contour detectiion
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv.dilate(img, kernel, iterations=1)

    # getting our numbers one by one
    rois = get_contour_bounding_rectangles_largest(img)
    # print(len(rois))
    grayd = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        nums.append((r[0], r[1], r[2], r[3]))
    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
    descs = []
    for i, r in enumerate(nums):
        if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
            continue
        im = cv.resize(img[r[1]:r[3], r[0]:r[2]], (32 * 10, 24 * 10), interpolation=cv.INTER_CUBIC)
        # blur = cv.GaussianBlur(im, (31, 31), 0)
        # ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        points = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], n_points)
        # points = sc.get_points_from_img(im, n_points)
        descriptor = sc.compute(points).flatten()
        descs.append(descriptor)
    # print(descs)
    return np.array(descs[0])
#
#
def match(base, current):
    """
      Here we are using cosine diff instead of "by paper" diff, cause it's faster
    """
    res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
    # res = cdist(base, current, metric="cosine")

    idxmin = np.argmin(res.reshape(len(base)))
    # result = sym_names[idxmin]
    return idxmin, res.reshape(len(base))[idxmin]
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
    user_set = ['ztx_cover', 'zx_cover']
    # symbol_set = ['qus', 'light']
    symbol_set = ['play', 'stop', 'qus', 'search', 'light']
    fsize1 = (32 * 10, 24 * 10)
    hu_size = (20 * 10, 20 * 10)
    HuM_symbol = defaultdict(list)
    symbol_ims = defaultdict(list)
    # read and process ref files
    ref_path = './heatlabel/refs/'
    filenames = [filename for filename in os.listdir(ref_path) if filename.endswith('.png')]
    imrefs = dict()
    cntrefs = dict()
    orbrefs = dict()
    surfrefs = dict()
    orb = cv.ORB_create()
    # surf = cv.xfeatures2d_SIFT.create()
    FLANN_INDEX_KDTREE = 0
    index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    FLANN_INDEX_LSH = 6
    index_params_orb = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)
    flann_orb = cv.FlannBasedMatcher(index_params_orb, search_params)
    flann_surf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    base_sc = list()
    symbols_sc = list()
    sc = ShapeContext()
    n_points = 50
    for filename in filenames:
        symbol = filename.split('.')[0]
        # if symbol not in symbol_set:
        #     continue
        imref = cv.imread(ref_path + filename, cv.IMREAD_GRAYSCALE)
        # imref = cv.resize(im, fsize1, interpolation=cv.INTER_CUBIC)
        imref = cv.GaussianBlur(imref, (11, 11), 0)
        _, imref = cv.threshold(imref, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        imref = cv.morphologyEx(imref, cv.MORPH_CLOSE, kernel)
        imrefs[symbol] = imref
        # orbrefs[symbol] = orb.detectAndCompute(imref, None)
        # # surfrefs[symbol] = surf.detectAndCompute(imref, None)
        # if orbrefs[symbol][1] is None:
        #     print('Error! No ORB descriptor found for {}'.format(symbol))
        # if surfrefs[symbol][1] is None:
        #     print('Error! No SURF descriptor found for {}'.format(symbol))
        # get contour
        contours, hierarchy = cv.findContours(imref, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours]
        if len(areas) == 0:
            continue
        max_index = np.argmax(areas)
        # print(areas[max_index], im.shape[0] * im.shape[1])
        cnt = contours[max_index]
        (x, y, w, h) = cv.boundingRect(cnt)
        cntrefs[symbol] = cnt
        #shape context
        base_sc.append(parse_img(imref[y:y+h, x:x+w], sc, n_points))
        symbols_sc.append(symbol)

        # plt.figure()
        # plt.imshow(imref[y:y+h, x:x+w])
        # plt.show()

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    X_sc = list()
    Y_sc = list()
    for user in user_set:
        path = './heatlabel/data/' + user + '/'
        filenames = [filename for filename in os.listdir(path) if filename.endswith('.jpg')]
        for filename in filenames:
            symbol = filename.split('_')[0]
            if symbol not in symbol_set:
                continue
            im = cv.imread(path + filename, cv.IMREAD_GRAYSCALE)
            # print(np.amax(im))
            im[im < 0.2*255] = 0
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
            imt = im[y:y + h, x:x + w]

            #shape context match
            recognize = parse_img(imt, sc, n_points)
            X_sc.append(recognize)
            Y_sc.append(symbol)
            idxrec, res = match(base_sc, recognize)
            print(symbol, symbols_sc[idxrec], res, len(recognize))
            # #orb shapematch
            # kpt, dest = orb.detectAndCompute(imt, None)
            # # kpt_surf, dest_surf = surf.detectAndCompute(imt, None)
            # if orbrefs[symbol][1] is None or dest is None:
            #     continue
            # matches = bf.knnMatch(orbrefs[symbol][1], dest, k=2)
            # # matches = bf.knnMatch(orbrefs[symbol][1], dest, k=2)
            # # matches = flann_surf.knnMatch(surfrefs[symbol][1], dest_surf, k=2)
            # if len(matches[0]) == 1:
            #     continue
            # img3 = cv.drawMatchesKnn(imrefs[symbol], orbrefs[symbol][0], imt, kpt, matches, imt, flags=2)
            # if symbol in 'play':
            #     plt.figure()
            #     plt.imshow(img3)
            #     plt.show()
            # good = list()
            # for (m,n) in matches:
            #     if m.distance < 0.75*n.distance:
            #         good.append(m)
            # print(symbol, len(good))
            # # hu's moments shapematch
            # dists = [cv.matchShapes(cnt, cntrefs[sym], cv.CONTOURS_MATCH_I1, 0) for sym in symbol_set]
            # idxmin = dists.index(min(dists))
            # print(symbol_set[idxmin], symbol, min(dists))

            # if (w * h) / (im.shape[0] * im.shape[1]) < 0.1:
            #     continue
            # th_crop = cv.resize(im[y:y + h, x:x + w], hu_size, interpolation=cv.INTER_CUBIC)
            # symbol_ims[symbol].append(im[y:y + h, x:x + w])
            # HuM_symbol[symbol].append(HuM(im[y:y + h, x:x + w]))
            # HuM_symbol[symbol].append(HuM(th_crop))
    #
    #
    # # create a ML model
    # X_train = list()
    # Y_train = list()
    # for k, v in HuM_symbol.items():
    #     for hv in v:
    #         # print(hv)
    #         X_train.append(hv)
    #         Y_train.append(k)
    X_train = X_sc
    Y_train = Y_sc
    svc = svm.SVC(gamma="scale")
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
    cv_ml = StratifiedKFold(2, random_state=1, shuffle=True)
    clf_svm = GridSearchCV(svc, parameters, cv=cv_ml)
    print(cross_val_score(clf_svm, X_train, Y_train, cv=cv_ml))
    clf_svm.fit(X_train, Y_train)

    clf_rf = RandomForestClassifier(n_estimators=2000, max_depth=9, random_state=0)
    scores_rf = cross_val_score(clf_rf, X_train, Y_train, cv=cv_ml)
    print("{0} RandomForest Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_rf, scores_rf.mean(), scores_rf.std() * 2))
    # clf_rf.fit(X_train, Y_train)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_train, Y_train, test_size=0.3, random_state=4)
    clf_rf.fit(X_train1, Y_train1)
    Y_predict = clf_rf.predict(X_test1)
    # acc_between.append(np.sum(Y_predict == Y_test)/ len(Y_test))
    cfm = confusion_matrix(Y_test1, Y_predict, labels=symbol_set)
    cfm_ratio = [list(item / np.sum(item)) for item in cfm]
    print(cfm_ratio)

    clf_gb = GradientBoostingClassifier()
    scores_gb = cross_val_score(clf_gb, X_train, Y_train, cv=cv_ml)
    print("{0} GradientBoosting Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_gb, scores_gb.mean(), scores_gb.std() * 2))