import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, TimeSeriesSplit
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupShuffleSplit
import sys
np.set_printoptions(threshold=sys.maxsize)

wine = datasets.load_wine()


# def custom_cv_2folds(X):
#     n = X.shape[0]
#     i = 1
#     while i <= 2:
#         idx = np.arange(n * (i-1)/2, n * i/2, dtype=int)
#         yield idx, idx
#         i += 1


X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.4,
    random_state=0
)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

metrics = ['accuracy']

# K-Fold
kf = KFold(n_splits=3)
generator = kf.split(wine)
# for train, test in kf.split(wine):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=kf)
print("walidacja k_fold")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=kf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Repeated K-Fold
random_state = 12883823
rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=random_state)
# for train, test in rkf.split(wine):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=rkf)
print("walidacja rkf")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=rkf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave One Out (LOO)
loo = LeaveOneOut()
# for train, test in loo.split(wine):
#     print("%s %s" % (train, test))
print("Walidacja loo:")
scores = cross_val_score(clf, wine.data, wine.target, cv=loo)
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=loo, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave P Out (LPO)
lpo = LeavePOut(p=2)
# for train, test in lpo.split(wine):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=lpo)
print("walidacja lpo")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=lpo, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Walidacja losowa permutacji krzyżowych Shuffle & Split
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
# for train_index, test_index in ss.split(wine):
#     print("%s %s" % (train_index, test_index))
scores = cross_val_score(clf, wine.data, wine.target, cv=ss)
print("walidacja ss")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=ss, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Iteratory walidacji krzyżowej ze stratyfikacją opartą na etykietach klas

# Stratified k-fold
skf = StratifiedKFold(n_splits=3)
# for train, test in skf.split(wine.data, wine.target):
#     print('train -  {}   |   test -  {}'.format(
#         np.bincount(wine.target[train]), np.bincount(wine.target[test])))
scores = cross_val_score(clf, wine.data, wine.target, cv=skf)
print("walidacja skf:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=skf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

kf = KFold(n_splits=3)
# for train, test in kf.split(wine.data, wine.target):
#     print('train -  {}   |   test -  {}'.format(
#         np.bincount(wine.target[train]), np.bincount(wine.target[test])))
scores = cross_val_score(clf, wine.data, wine.target, cv=kf)
print("walidacja kf2:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=kf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Iteratory w, walidacji krzyżowej dla zgrupowanych danych
# Grupowanie k-fold
X = wine.data
y = wine.target
groups = np.rint(wine.data[:, 0])
gkf = GroupKFold(n_splits=3)
# for train, test in gkf.split(X, y, groups=groups):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=gkf, groups=groups)
print("walidacja gkf:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=gkf, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=3)
# for train, test in sgkf.split(X, y, groups=groups):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=sgkf, groups=groups)
print("walidacja sgkf")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=sgkf, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave One Group Out
logo = LeaveOneGroupOut()
# for train, test in logo.split(X, y, groups=groups):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=logo, groups=groups)
print("walidacja logo")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=logo, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave P Groups Out
lpgo = LeavePGroupsOut(n_groups=2)
# for train, test in lpgo.split(X, y, groups=groups):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=lpgo, groups=groups)
print("walidacja lpgo")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=lpgo, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Group Shuffle Split
gss = GroupShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
# for train, test in gss.split(X, y, groups=groups):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=gss, groups=groups)
print("walidacja gss")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=gss, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Time Series Split
tscv = TimeSeriesSplit(n_splits=4)
print(tscv)
# for train, test in tscv.split(wine):
#     print("%s %s" % (train, test))
scores = cross_val_score(clf, wine.data, wine.target, cv=tscv)
print("walidacja tscv", scores)
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



