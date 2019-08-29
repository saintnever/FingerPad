import numpy as np
import pickle
import os
import scipy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.metrics import confusion_matrix

path = './handwriting/'
filenames = [filename for filename in os.listdir(path) if filename.endswith('dir.pkl')]

X = list()
Y = list()
y_unique = ['check', 'cross', 'charA', 'charZ']
n = 18
for filename in filenames:
    with open(path+filename, 'rb') as file:
        gesture_raw, gesture_kalman = pickle.load(file)
    gesture_set = gesture_kalman
    result = filename.split('_')[0]
    if result[-2:] == '50':
        result = result[:-2]
    if result == 'charT' or result == 'charA':
        continue
    # y_true = y_unique.index(result)
    for gesture in gesture_set:
        gesture = np.array(gesture).transpose()
        angle = gesture[0]
        amplitude = gesture[1]
        # print(max(amplitude), min(amplitude), np.mean(amplitude))
        angle_large = list()
        # for i in range(n):
        #     for j, ang in enumerate(angle)
        for idx, amp in enumerate(amplitude):
            # if amp > np.mean(amplitude) * 0.1:
            if amp > 0.7:
            # if amp > np.quantile(amplitude, 0.25):
                angle_large.append(angle[idx])
        # for i in range(-2, 2):
        #     angle_rotate = np.array(angle_large) + i
        #     for j, a in enumerate(angle_rotate):
        #         if a > 180:
        #             angle_rotate[j] = a - 360
        #         if a < -180:
        #             angle_rotate[j] = 360 + a
            # angle_rotate[angle_rotate >= 360] = angle_rotate[angle_rotate >= 360] - 360
            # angle_rotate = [180 - angle for angle in angle_rotate if angle > 180 else angle]
        hist, avg = np.histogram(angle_large, n, density=False)
        x = hist/np.sum(hist)
        # x = hist
        # angle_shift = int(20/(360/n))
        # for i in range(-angle_shift,angle_shift):
        # X.append(np.roll(x,i))
        X.append(x)
        Y.append(result)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=4)

parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=3)
# clf = svm.SVC(kernel='poly', gamma='scale')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
cfm = confusion_matrix(Y_test, Y_predict, labels=['check', 'cross', 'charZ'])
cfm = cfm / np.sum(cfm, axis=1)
print('CF Maxtrix is {}, and accuracy is {}'.format(cfm, np.mean(np.diagonal(cfm))))

clf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
cv = StratifiedKFold(3, random_state=1, shuffle=True)
scores = cross_val_score(clf_svm, X, Y, cv=cv)
print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores, scores.mean(), scores.std() * 2))

# clf_rf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=0)
# # clf_rf.fit(X, Y)
# scores_rf = cross_val_score(clf_rf, X, Y, cv=cv)
# print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_rf, scores_rf.mean(), scores_rf.std() * 2))

clf_gb = GradientBoostingClassifier()
# clf_gb.fit(X, Y)
scores_gb = cross_val_score(clf_gb, X, Y, cv=cv)
print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(scores_gb, scores_gb.mean(), scores_gb.std() * 2))

# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gb.mean(), scores_gb.std() * 2))
