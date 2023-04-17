import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, TimeSeriesSplit
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
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

X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.4,
    random_state=0
)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

metrics = ['accuracy']

# K-Fold
kf = KFold(n_splits=3)  # model walidaji k-fold, dzielimy na 3 foldy, bo wiemy, że są 3 klasy
generator = kf.split(wine)  # rozdzielenie danych dot. win
scores = cross_val_score(clf, wine.data, wine.target, cv=kf)  # testowanie modelu za pomocą cross_val_score
# wine_data to parametry opisujące dane wino, natomisat wine.target to etykieta (klasa) wina
print("Walidacja k_fold:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # cross_val score zwraca tablicę wyników,
# zatem musimy wyliczyć z nich średnią oraz odchylenie standardowe
scores2 = cross_validate(clf, wine.data, wine.target, cv=kf, scoring=metrics)  # testowanie modelu za pomocą
# cross_validate, metrics jest 'accuracy' - pozostałe parametry analogicznie
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))
# cross_validate zwraca dictionary, zatem należy "wyciągnąć" z tego wynik badania dokładności 'test_accuracy'
# i dopiero z tego wyznaczyć średnią i odchylenie standardowe

# Repeated K-Fold
random_state = 12883823  # tu wstawiamy dowolnego int'a, w przeciwnym razie algorytm sam wygeneruje tę wartość
# i możemy za każdym razem otrzymać inny wynik testu z uwagi na inny podział grupy treningowej/testowej
rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=random_state)
# wszytko inne analogicznie jak w k-fold, jedynie inny model
scores = cross_val_score(clf, wine.data, wine.target, cv=rkf)
print("walidacja rkf")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=rkf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave One Out (LOO)
loo = LeaveOneOut()  # ten model potrzebnych w tym momencie konkretnych parametrów nie ma
print("Walidacja loo:")
scores = cross_val_score(clf, wine.data, wine.target, cv=loo)
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=loo, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Leave P Out (LPO)
lpo = LeavePOut(p=2)  # 2 to wielkosc zestawu testowego
scores = cross_val_score(clf, wine.data, wine.target, cv=lpo)
print("walidacja lpo")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=lpo, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Walidacja losowa permutacji krzyżowych Shuffle & Split
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)  # 5 iteracji, dzielmy zestaw danych
# na 75% treningowych i 25% testowych
scores = cross_val_score(clf, wine.data, wine.target, cv=ss)
print("walidacja ss")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=ss, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))


# Iteratory walidacji krzyżowej ze stratyfikacją opartą na etykietach klas

# Stratified k-fold
skf = StratifiedKFold(n_splits=3)
scores = cross_val_score(clf, wine.data, wine.target, cv=skf)
print("walidacja skf:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=skf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

kf = KFold(n_splits=3)
scores = cross_val_score(clf, wine.data, wine.target, cv=kf)
print("walidacja kf2:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=kf, scoring=metrics)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# Iteratory walidacji krzyżowej dla zgrupowanych danych

# Grupowanie k-fold
X = wine.data  # podzaiał danych na parametry oraz etykiety, stosowany tylko do drukowania wyników
y = wine.target
groups = np.rint(wine.data[:, 0])  # dane musimy zgrupować, tu grupowano wg. jednej z kolumn
# po zaokrągleniu do liczny całkowitej
gkf = GroupKFold(n_splits=3)
scores = cross_val_score(clf, wine.data, wine.target, cv=gkf, groups=groups)  # w ocenie musimy już wziąć pod uwagę
# podział na grupy
print("walidacja gkf:")
print("Accuracy %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores2 = cross_validate(clf, wine.data, wine.target, cv=gkf, scoring=metrics, groups=groups)
print("Accuracy %0.2f (+/- %0.2f)" % (scores2['test_accuracy'].mean(), scores2['test_accuracy'].std() * 2))

# StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=3)

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
gss = GroupShuffleSplit(n_splits=3, test_size=0.5, random_state=0)  # dzielimy zestaw dancyh na treningowe
# i testowe pół na pół
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
