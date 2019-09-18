import numpy as np
import pickle
import os
import scipy
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

user_set = ['txz2', 'zx2', 'dxy', 'cjn', 'hn', 'gtt']
# symbol_set = ['back', 'cross', 'tick', 'ques', 'CA', 'two']
symbol_set = ['ac', 'tv', 'l1', 'l2', 'k', 'p', 'a', 't', 'bs']
classname = ['AC', 'TV', 'L1',  'L2', 'K', 'P', 'A', 'T', '$\leftarrow$']

X_uset = []
Y_uset = []
n = 12
for user in user_set:
    path = './handwriting/'+user+'/processed/'
    filenames = [filename for filename in os.listdir(path) if filename.endswith('_dir.pkl')]
    X = list()
    Y = list()
    for filename in filenames:
        result = filename.split('_')[1]
        if len(symbol_set) > 0 and result not in symbol_set:
            continue
        with open(path + filename, 'rb') as file:
            [gesture_raw, gesture_kalman] = pickle.load(file)
        if len(gesture_kalman[-1][0]) > 2:
            gesture_set = gesture_kalman[:-1] + gesture_kalman[-1]
        else:
            gesture_set = gesture_kalman
        for gesture in gesture_set:
            angle_raw = [item[0] for item in gesture]
            amplitude = [item[1] for item in gesture]
            angle = list()
            amp_filtered = list()
            for i, amp in enumerate(amplitude):
                if amp > 0.2 * np.mean(amplitude):
                # if amp > 0:
                    angle.append(angle_raw[i])
                    amp_filtered.append(amp)
            # print(max(amplitude), min(amplitude), np.mean(amplitude))
            # amp_hist = [[] for _ in range(n)]
            angle_hist = [[] for _ in range(n)]
            angle_first = angle[:int(len(angle) / 2)]
            angle_second = angle[int(len(angle) / 2):]
            hist_first, avgf = np.histogram(angle_first, n, range=(-180, 180), density=False)
            hist_second, avgs = np.histogram(angle_second, n, range=(-180, 180), density=False)
            hist, avg = np.histogram(angle, n, range=(-180, 180), density=False)
            x_first = hist_first / np.sum(hist_first)
            x_second = hist_second/np.sum(hist_second)
            x = hist / np.sum(hist)
            amp_hist = [0 for _ in hist]
            amp_second_hist = [0 for _ in hist]
            amp_first_hist = [0 for _ in hist]
            amp_first = amp_filtered[:int(len(amp_filtered) / 2)]
            amp_second = amp_filtered[int(len(amp_filtered) / 2):]
            for i, amp in enumerate(amp_filtered):
                diff = angle_raw[i] - np.array(avg)
                diff[diff < 0] = 360
                index = np.argmin(diff)
                amp_hist[index] += amp
            for i, amp in enumerate(amp_first):
                diff = angle_first[i] - np.array(avg)
                diff[diff < 0] = 360
                index = np.argmin(diff)
                amp_first_hist[index] += amp
            for i, amp in enumerate(amp_second):
                diff = angle_second[i] - np.array(avg)
                diff[diff < 0] = 360
                index = np.argmin(diff)
                amp_second_hist[index] += amp
            amp_hist = amp_hist / np.sum(amp_hist)
            amp_first_hist = amp_first_hist / np.sum(amp_first_hist)
            amp_second_hist = amp_second_hist / np.sum(amp_second_hist)
            # angle_shift = int(20/(360/n))
            # for i in range(-angle_shift,angle_shift):
            # X.append(np.roll(x,i))
            # X.append(list(x_first) + list(x_second) + list(amp_hist))
            X.append(list(x_first) + list(x_second) + list(amp_first_hist) + list(amp_second_hist))
            # X.append(list(amp_first_hist) + list(amp_second_hist))
            # X.append(list(x_first) + list(x_second))
            # X.append(x + list(amp_hist))
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

# Overall-test
X_total = list()
Y_total = list()
for i in range(len(X_uset)):
    for j in range(len(X_uset[i])):
        X_total.append(X_uset[i][j])
        Y_total.append(Y_uset[i][j])

# plot the features for two gestures
feature_ac = X_total[0]
feature_tv = X_total[421]
fig, ax = plt.subplots(2, 1,sharex='all', sharey='all')
ax[0].bar(range(len(feature_ac)), feature_ac)
# plt.show()
ax[1].bar(range(len(feature_tv)), feature_tv)
plt.xlim([0,47])
plt.show()

np.set_printoptions(precision=2)
cv = StratifiedKFold(3, random_state=1, shuffle=True)
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.3, random_state=4)
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=cv)
print(clf)
# clf = svm.SVC(kernel='poly', gamma='scale')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
cfm_ratio = [list(item/np.sum(item)) for item in cfm]
# df_cm = pd.DataFrame(cfm_ratio, classname, classname)
# sn.set(font_scale=1.4)
# sn.heatmap(df_cm, annot=True, annot_kws={'size':14}, fmt='.2f', cmap="YlGnBu")
# plt.show()
print('CF Maxtrix is {0}, and accuracy is {1:.3f}'.format(cfm_ratio, np.mean(np.diagonal(cfm_ratio))))

clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
# cv = StratifiedKFold(3, random_state=1, shuffle=True)
scores = cross_val_score(clf_svm, X_total, Y_total, cv=cv)
print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores, scores.mean(), scores.std() * 2))

# clf_rf = RandomForestClassifier(n_estimators=3000, max_depth=9, random_state=0)
# # clf_rf.fit(X, Y)
# scores_rf = cross_val_score(clf_rf, X_total, Y_total, cv=cv)
# print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_rf, scores_rf.mean(), scores_rf.std() * 2))
#
# clf_gb = GradientBoostingClassifier()
# scores_gb = cross_val_score(clf_gb, X_total, Y_total, cv=cv)
# print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_gb, scores_gb.mean(), scores_gb.std() * 2))

# within user accuracy
acc_within = list()
cv = StratifiedKFold(2, random_state=1, shuffle=True)
acc_matrix = np.zeros((len(symbol_set), len(symbol_set)))
for i in range(len(X_uset)):
    X_total = X_uset[i]
    Y_total = Y_uset[i]
    clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    # acc_within.append(np.mean(cross_val_score(clf_svm, X_total, Y_total, cv=cv)))
    X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.3, random_state=4)
    clf_svm.fit(X_train, Y_train)
    Y_predict = clf_svm.predict(X_test)
    acc_within.append(np.sum(Y_predict == Y_test)/ len(Y_test))
    cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
    cfm_ratio = [list(item / np.sum(item)) for item in cfm]
    acc_matrix += cfm_ratio
    print('{} within accuracy is {}'.format(user_set[i], acc_within[-1]))

print('Averaged within user accuracy is {}, std is {}'.format(np.mean(acc_within), np.std(acc_within)))
df_cm = pd.DataFrame(acc_matrix/len(user_set), classname, classname)
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={'size':14}, fmt='.2f', cmap="YlGnBu")
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
    clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf_svm.fit(X_train, Y_train)
    Y_predict = clf_svm.predict(X_test)
    # acc_between.append(np.sum(Y_predict == Y_test)/ len(Y_test))
    cfm = confusion_matrix(Y_test, Y_predict, labels=symbol_set)
    cfm_ratio = [list(item / np.sum(item)) for item in cfm]
    acc_matrix += cfm_ratio
    # print(cfm_ratio)
    acc_between.append(clf_svm.score(X_test, Y_test))
    print('{} between accuracy is {}'.format(user_set[i], acc_between[-1]))
print('Averaged between user accuracy is {}, std is {}'.format(np.mean(acc_between), np.std(acc_between)))
print('Averaged between user confusion matrix is {}'.format(acc_matrix/len(user_set)))
df_cm = pd.DataFrame(acc_matrix/len(user_set), classname, classname)
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={'size':14}, fmt='.2f', cmap="YlGnBu")
plt.show()


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

