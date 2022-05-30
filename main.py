import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pickle
import seaborn as sns

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv("types.csv")
print('**************Dataset Head**************')
print(df.head())
print('**************Dataset tail**************')
print(df.tail())

print('**************Datatype**************')
print(df.dtypes)







X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state = 2000)

clf.fit(X_train,y_train)

prediction_test = clf.predict(X_test)

#print(prediction_test)
from sklearn import metrics
print('**************Random Forest**************')
print(f"Accuracy of Random Forest :  {metrics.accuracy_score(y_test,prediction_test)* 100:.2f}% ")
precision = precision_score(y_test, prediction_test , average='weighted')
recall = recall_score(y_test, prediction_test, average='weighted')
f1 = f1_score(y_test, prediction_test  , average='weighted')
cm = metrics.confusion_matrix(y_test, prediction_test)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1-score")
print("%.3f" %f1)
print("CONFUSION MATRIX")
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])


#feature_importances
feature_list = list(X.columns)
feature_imp = pd.Series(clf.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_test)

plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix of RF - score:'+str(accuracy_score(y_test,prediction_test))
plt.title(all_sample_title, size = 15);
plt.show()

pickle.dump(clf, open('model.pkl', 'wb'))

model=pickle.load(open('model.pkl', 'rb'))

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
print('**************KNN**************')
print(f"Accuracy of KNN :  {metrics.accuracy_score(y_test,y_pred)* 100:.2f}% ")
precision = precision_score(y_test, y_pred , average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred  , average='weighted')
cm = metrics.confusion_matrix(y_test, y_pred)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1-score")
print("%.3f" %f1)
print("CONFUSION MATRIX")
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
