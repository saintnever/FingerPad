import numpy as np
import pickle
import os
import scipy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.metrics import confusion_matrix

user_set = ['ztx', 'yy']
symbol_set = ['charA', 'charZ', 'check', 'charX']
X_uset = []
Y_uset = []
n = 18
for user in user_set:
    path = './handwriting/'+user+'/processed/'
    filenames = [filename for filename in os.listdir(path) if filename.endswith('_dir.pkl')]
    X = list()
    Y = list()
    for filename in filenames:
        result = filename.split('_')[1]
        if result not in symbol_set:
            continue
        with open(path + filename, 'rb') as file:
            [gesture_raw, gesture_kalman] = pickle.load(file)
        gesture_set = gesture_kalman
        # y_true = y_unique.index(result)
        for gesture in gesture_set:
            gesture = np.array(gesture).transpose()
            angle = gesture[0]
            amplitude = gesture[1]
            # print(max(amplitude), min(amplitude), np.mean(amplitude))
            amp_hist = [[] for _ in range(n)]
            angle_hist = [[] for _ in range(n)]
            angle_first = angle[:int(len(angle) / 2)]
            angle_second = angle[int(len(angle) / 2):]
            hist_first, avgf = np.histogram(angle_first, n, range=(-180, 180), density=False)
            hist_second, avgs = np.histogram(angle_second, n, range=(-180, 180), density=False)
            x_first = hist_first / np.sum(hist_first)
            x_second = hist_second/np.sum(hist_second)
            # angle_shift = int(20/(360/n))
            # for i in range(-angle_shift,angle_shift):
            # X.append(np.roll(x,i))
            X.append(list(x_first)+list(x_second))
            Y.append(result)
            # manual calculate histogram
            # for i in range(n):
            #     for j, ang in enumerate(angle):
            #         if -180 + i*360/n <= ang < -180 + (i+1)*360/n:
            #             angle_hist[i].append(ang)
            #             amp_hist[i].append(amplitude[j])
            # x_ang = [0 for _ in range(n)]
            # x_amp = [0 for _ in range(n)]
            # for i in range(n):
            #     x_ang[i] = len(angle_hist[i])
                # x_amp[i] = np.mean(amp_hist[i])
            # x_ang = x_ang/np.sum(x_ang)
            # hist, avg = np.histogram(angle, bins=n, range=(-180, 180), density=False)
            # print(x_ang, hist)
            # rotate to account for input angle variation
            # for i in range(-2, 2):
            #     angle_rotate = np.array(angle_large) + i
            #     for j, a in enumerate(angle_rotate):
            #         if a > 180:
            #             angle_rotate[j] = a - 360
            #         if a < -180:
            #             angle_rotate[j] = 360 + a
                # angle_rotate[angle_rotate >= 360] = angle_rotate[angle_rotate >= 360] - 360
                # angle_rotate = [180 - angle for angle in angle_rotate if angle > 180 else angle]
    X_uset.append(X)
    Y_uset.append(Y)

X_total = list()
Y_total = list()
for i in range(len(X_uset)):
    for j in range(len(X_uset[i])):
        X_total.append(X_uset[i][j])
        Y_total.append(Y_uset[i][j])

X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.33, random_state=4)
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=3)
# clf = svm.SVC(kernel='poly', gamma='scale')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
cfm_ratio = [list(item/np.sum(item)) for item in cfm]
print('CF Maxtrix is {0}, and accuracy is {1:.3f}'.format(cfm_ratio, np.mean(np.diagonal(cfm_ratio))))

clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf_svm.fit(X_uset[1], Y_uset[1])
print('Between user accuracy {}'.format(clf_svm.score(X_uset[0], Y_uset[0])))

cv = StratifiedKFold(3, random_state=1, shuffle=True)
scores = cross_val_score(clf_svm, X_total, Y_total, cv=cv)
print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores, scores.mean(), scores.std() * 2))

# clf_rf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=0)
# # clf_rf.fit(X, Y)
# scores_rf = cross_val_score(clf_rf, X, Y, cv=cv)
# print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_rf, scores_rf.mean(), scores_rf.std() * 2))

clf_gb = GradientBoostingClassifier()
scores_gb = cross_val_score(clf_gb, X_total, Y_total, cv=cv)
print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_gb, scores_gb.mean(), scores_gb.std() * 2))

#
# # print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gb.mean(), scores_gb.std() * 2))
# filenames = [filename for filename in os.listdir(path1) if filename.endswith('dir.pkl')]
#
# X = list()
# Y = list()
# y_unique = ['check', 'charX', 'charA']
# n = 18
# for filename in filenames:
#     with open(path1+filename, 'rb') as file:
#         gesture_raw, gesture_kalman = pickle.load(file)
#     gesture_set = gesture_kalman
#     result = filename.split('_')[1]
#     if result[-2:] == '50':
#         result = result[:-2]
#     if result == 'charV' or result == 'charZ':
#         continue
#     # y_true = y_unique.index(result)
#     for gesture in gesture_set:
#         gesture = np.array(gesture).transpose()
#         angle = gesture[0]
#         amplitude = gesture[1]
#         # print(max(amplitude), min(amplitude), np.mean(amplitude))
#         amp_hist = [[] for _ in range(n)]
#         angle_hist = [[] for _ in range(n)]
#
#         for i in range(n):
#             for j, ang in enumerate(angle):
#                 if -180 + i*360/n <= ang < -180 + (i+1)*360/n:
#                     angle_hist[i].append(ang)
#                     amp_hist[i].append(amplitude[j])
#         x_ang = [0 for _ in range(n)]
#         x_amp = [0 for _ in range(n)]
#         for i in range(n):
#             x_ang[i] = len(angle_hist[i])
#             # x_amp[i] = np.mean(amp_hist[i])
#         # x_ang = x_ang/np.sum(x_ang)
#         # hist, avg = np.histogram(angle, bins=n, range=(-180, 180), density=False)
#         # print(x_ang, hist)
#         # for i in range(-2, 2):
#         #     angle_rotate = np.array(angle_large) + i
#         #     for j, a in enumerate(angle_rotate):
#         #         if a > 180:
#         #             angle_rotate[j] = a - 360
#         #         if a < -180:
#         #             angle_rotate[j] = 360 + a
#             # angle_rotate[angle_rotate >= 360] = angle_rotate[angle_rotate >= 360] - 360
#             # angle_rotate = [180 - angle for angle in angle_rotate if angle > 180 else angle]
#         hist, avg = np.histogram(angle, n, range=(-180, 180), density=False)
#         x = hist/np.sum(hist)
#         # angle_shift = int(20/(360/n))
#         # for i in range(-angle_shift,angle_shift):
#         # X.append(np.roll(x,i))
#         X.append(x_ang)
#         Y.append(result)
#
# # clf_svm.score(X, Y)
# Y_predict = clf_svm.predict(X)
# cfm = confusion_matrix(Y, Y_predict, labels=y_unique)
# cfm = cfm / np.sum(cfm, axis=1)
# print('CF Maxtrix is {}, and accuracy is {}'.format(cfm, np.mean(np.diagonal(cfm))))

# path = './handwriting/ztx/processed/'
# filenames = [filename for filename in os.listdir(path) if filename.endswith('dir.pkl')]
#
# for filename in filenames:
#     with open(path+filename, 'rb') as file:
#         gesture_raw, gesture_kalman = pickle.load(file)
#
#     gesture_set = gesture_kalman
#     result = filename.split('_')[1]
#     if result == 'charT' or result[-2:] == '50':
#         continue
#     else:
#         fparts = filename.split('_')
#         fname = fparts[0]+'_'+ fparts[1] + '50_' + fparts[2]
#         with open(path + fname, 'rb') as file:
#             gesture_raw_more, gesture_kalman_more = pickle.load(file)
#         gesture_raw.append(gesture_raw_more)
#         gesture_kalman.append(gesture_kalman_more)
#         with open(path + filename+'100', 'wb') as file:
#             pickle.dump([gesture_raw, gesture_kalman], file)

