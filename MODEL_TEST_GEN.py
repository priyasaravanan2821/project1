#Import Libraries
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Load data and Check Data
inf = pd.read_csv('PCOS_data.csv')

#change PCOS(Y/N) to Target
data = inf.rename(columns = {"PCOS (Y/N)":"Target"})


data.head()

# Dropping unnecessary features.
data = data.drop(["Sl. No","Patient File No."],axis = 1)
data.info(verbose = True, null_counts = False)

#Let's look at the dtype is an object
data["AMH(ng/mL)"].head()
data["II    beta-HCG(mIU/mL)"].head()

#As you can see some numeric data is saved as strings : AMH(ng/mL) , II beta-HCG(mIU/mL). Let's converting them.
#Converting
data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')


colors = ['#002667','#3c0067']


def bar_plot(variable):

    # get feature
    var = data[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    # visualize
    plt.figure(figsize=(9, 3))
    plt.bar(varValue.index, varValue, color=colors)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Count")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable, varValue))


category = ["Target", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)",
            "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", "Blood Group"]
for c in category:
    bar_plot(c)


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(data[variable], bins = 50,color=colors[0])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()

numericVar = ["Age (yrs)", "Weight (Kg)","Marraige Status (Yrs)"]
for n in numericVar:
    plot_hist(n)


data.columns[data.isnull().any()]

#Filling missing values with the median value of the features.

data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(),inplace=True)
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(),inplace=True)
data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median(),inplace=True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace=True)

data.isnull().sum()


#   Feature Extraction
data.describe()

corr_matrix= data.corr()
plt.subplots(figsize=(30,10))
sns.heatmap(corr_matrix,cmap="Pastel1", annot = True, fmt = ".2f");
plt.title("Correlation Between Features")
plt.show()

threshold = 0.25
filtre = np.abs(corr_matrix["Target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
plt.subplots(figsize=(10,7))
sns.heatmap(data[corr_features].corr(),cmap="Pastel1", annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Theshold 0.25")
plt.show()

#Assiging the features (X)and target(y).

X= data.drop(labels = ["Target","Pulse rate(bpm)","RR (breaths/min)","Hb(g/dl)", "Marraige Status (Yrs)","FSH(mIU/mL)","LH(mIU/mL)","FSH/LH","TSH (mIU/L)","AMH(ng/mL)","PRL(ng/mL)","Vit D3 (ng/mL)","PRG(ng/mL)","RBS(mg/dl)","BP _Systolic (mmHg)","BP _Diastolic (mmHg)","Follicle No. (L)","Follicle No. (R)","Avg. F size (L) (mm)","Avg. F size (R) (mm)","Endometrium (mm)","  I   beta-HCG(mIU/mL)","II    beta-HCG(mIU/mL)"],axis = 1)
y=data.Target

#Splitting the data into test and training sets.

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2)
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Accuracy of Logistic Regression: {} %  ".format(acc_log_train))


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {'max_depth' : [4,5,6,7,8,9,10,12]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}


classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(round(clf.best_score_*100,2))
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])

best_estimators

dt = best_estimators[0]
svm = best_estimators[1]
rf = best_estimators[2]
lr = best_estimators[3]
knn = best_estimators[4]

model_list = ['Decision Tree','SVC','rf','Logistic Regression','KNearestNeighbours']

fg = sns.factorplot(x = model_list, y = cv_result, size= 6, aspect=2 ,color= colors[1], saturation=5,kind='bar', data=data)
plt.title('Accuracy of different Classifier Models')
plt.xlabel('Classifier Models')
plt.ylabel('% of Accuracy')

plt.show()
import plotly.graph_objects as go
# create trace1
trace1 = go.Bar(
                x = model_list,
                y = cv_result,
                marker = dict(color = 'rgb(0, 128, 128)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(title = 'Accuracy of different Classifier Models' , xaxis = dict(title = 'Classifier Models'), yaxis = dict(title = '% of Accuracy'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

model = [dt,svm,rf,lr,knn]
predictions = []

for i in model:
    predictions.append(i.predict(X_test))
for j in range(5):
    print(" {} Accuracy :".format(model_list[j]))
    print(f' {round(metrics.accuracy_score(y_test, predictions[j]) * 100, 2)}%')
    print( metrics.accuracy_score(y_test, predictions[j]))
    print(" Precision :".format(model_list[j]))
    print(metrics.precision_score(y_test, predictions[j]))
    print(" recall :".format(model_list[j]))
    print(metrics.recall_score(y_test, predictions[j]))
    print(" f1_score :".format(model_list[j]))
    print(metrics.f1_score(y_test, predictions[j]))
    cm = confusion_matrix(y_test, predictions[j])
    plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Pastel1)
    plt.title(" {} Confusion Matrix".format(model_list[j]))
    plt.xticks(range(2), ["Not Pcos","Pcos"], fontsize=16)
    plt.yticks(range(2), ["Not Pcos","Pcos"], fontsize=16)
    plt.show()

import pickle

# Saving model to disk
pickle.dump(rf, open('model log rf.pkl','wb'))

