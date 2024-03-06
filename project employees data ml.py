import pandas as pd
data=pd.read_csv("C:/Users/saiga/Downloads/ANN-master/ANN-master/employees dataset.csv")
data.dtypes
data.columns
sum(data.duplicated())
data.isna().sum()
import seaborn as sns
# winsorization
sns.boxplot(data.salary_in_usd)
IQR = data['salary_in_usd'].quantile(0.75)-data['salary_in_usd'].quantile(0.25)
lower_limit = data['salary_in_usd'].quantile(0.75)-(IQR*1.5)
upper_limit = data['salary_in_usd'].quantile(0.25)+(IQR*1.5)
import numpy as np
outliers_data = np.where(data['salary_in_usd']> upper_limit, True, np.where(data['salary_in_usd']< lower_limit,True,False))
sum(outliers_data)

data_trimmed = data.loc[(~outliers_data,)]
data.shape
data_trimmed.shape
sns.boxplot(data_trimmed.salary_in_usd)
df=data_trimmed.reset_index(drop=True)


# AUTO EDA
pip install ydata-profiling
import os
from ydata_profiling import ProfileReport
report = ProfileReport(df,explorative=True)
report
report.to_file("EDA_report.html")

os.getcwd()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['job_title']=l.fit_transform(df['job_title'])
df['experience_level']=l.fit_transform(df['experience_level'])
df['employment_type']=l.fit_transform(df['employment_type'])
df['work_models']=l.fit_transform(df['work_models'])
df['employee_residence']=l.fit_transform(df['employee_residence'])
df['company_location']=l.fit_transform(df['company_location'])
df['company_size']=l.fit_transform(df['company_size'])

df.columns
df.drop(['salary','salary_currency'],axis=1,inplace=True)
df.columns

x=df.iloc[:,:8]
y=df.iloc[:,-1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
lr.accuracy=accuracy_score(y_test,y_pred)
lr.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=lr.score(x_train,y_train)
train_score
test_score=lr.score(x_test,y_test)
test_score


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit( x_train, y_train)
y_pred = dt.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
dt.accuracy = accuracy_score(y_test,y_pred)
dt.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=dt.score(x_train,y_train)
train_score
test_score=dt.score(x_test,y_test)
test_score

#SVC
from sklearn.svm import SVC
s=SVC()
s.fit(x_train,y_train)
y_pred=s.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
s.accuracy = accuracy_score(y_test,y_pred)
s.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=s.score(x_train,y_train)
train_score
test_score=s.score(x_test,y_test)
test_score

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
rf.accuracy = accuracy_score(y_test,y_pred)
rf.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=rf.score(x_train,y_train)
train_score
test_score=rf.score(x_test,y_test)
test_score

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
k=3
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
knn.accuracy = accuracy_score(y_test,y_pred)
knn.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=knn.score(x_train,y_train)
train_score
test_score=knn.score(x_test,y_test)
test_score

#GaussianNB
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
nb.accuracy = accuracy_score(y_test,y_pred)
nb.accuracy
print(classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix
train_score=nb.score(x_train,y_train)
train_score
test_score=nb.score(x_test,y_test)
test_score


#ensemble booster
import numpy as np
from sklearn.ensemble import BaggingClassifier

# Bagging with Decision Trees
bagging_classifier = BaggingClassifier(base_estimator=RandomForestClassifier(),n_estimators=10,random_state=42)
bagging_classifier.fit(x_train, y_train)
y_pred_bagging= bagging_classifier.predict(x_test)
y_pred
accuracy_bagging=accuracy_score(y_test, y_pred_bagging)
accuracy_bagging



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic data without clusters
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1,n_redundant=0, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers={
    'lr':LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    's': SVC(),
    'rf': RandomForestClassifier(),
    'knn': KNeighborsClassifier(n_neighbors=k),
    'bagging_classifier':BaggingClassifier()
}

# Plot decision boundaries and show accuracy
plt.figure(figsize=(12, 4))
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    plt.subplot(2, 3, i + 1)
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.title(name)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

