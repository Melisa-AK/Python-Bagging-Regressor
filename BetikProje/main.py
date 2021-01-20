import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#warning library
import warnings
warnings.filterwarnings('ignore')

#studentAssessment.csv dosyasının eklenmesi
data=pd.read_csv('studentAssessment.csv')

#Dosya içerisindeki NaN satır olup olmadığını kontrol ediyoruz. Ayrıca dosya bilgisini de görmüş oluyoruz.
data.info()

#Dosya içeriği:
print(data)
print("-------------------------------------------------------")


#is_banked değerlerinde kaç adet 1 ve 0 olduğu görüntülenir:
sns.countplot(data["is_banked"])
print(data.is_banked.value_counts())
plt.show()


#bağımlı değişken y, bağımsız değişkenler x'e atanır
X = data.iloc[:, [0,1]].values
y = data.iloc[:, 3].values


#Veri seti Yüzde 70’i eğitim seti yüzde 30’u da test seti olarak ayrılmıştır.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


#Karar Ağacı Modeli Eğitilmesi
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
karar_agaci= DecisionTreeClassifier()
karar_agaci.fit(X_train, y_train)
y_pred=karar_agaci.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print("-------------------------------------------------------")
print(confusion_matrix(y_test, y_pred))
print("Decision Doğruluk Oranı: ", accuracy_score(y_test, y_pred))


#Random Forest Modelinin Oluşturulması
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators=15, random_state=0)
randomForest.fit(X_train, y_train)
y_pred = randomForest.predict(X_test)
print("-------------------------------------------------------")
print(confusion_matrix(y_test, y_pred))
print("Random Doğruluk Oranı: ", accuracy_score(y_test, y_pred))


#Bagging Modelinin Oluşturulması
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(DecisionTreeClassifier(),
                            max_samples = 0.5,
                            max_features = 1.0, n_estimators = 20)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print("-------------------------------------------------------")
print(confusion_matrix(y_test, y_pred))
print("Bagging Doğruluk Oranı: ", accuracy_score(y_test, y_pred))



#Veri Yükleme ve SVC Eğit
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
y = y == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
svc = SVC(random_state=1)
svc.fit(X_train, y_train)

#ROC Eğrisinin Çizilmesi
svc_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()

##################################################
#Karar Ağacı ve ROC eğrisi
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
ax = plt.gca()
dtc_disp = plot_roc_curve(dtc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8, linestyle='-.')
plt.show()
##################################################
#Rastgele Orman eğitimi ve ROC eğrisi
rfc = RandomForestClassifier(n_estimators=15, random_state=1)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8, linestyle='-.')
plt.show()

############################################
bagc = BaggingClassifier(DecisionTreeClassifier(),
                            max_samples = 0.5,
                            max_features = 1.0, n_estimators = 20)
bagc.fit(X_train, y_train)
y_pred = bagc.predict(X_test)
ax = plt.gca()
bagc_disp = plot_roc_curve(bagc, X_test, y_test, ax=ax, alpha=0.6)
svc_disp.plot(ax=ax, alpha=0.8, linestyle='-.')

plt.show()
