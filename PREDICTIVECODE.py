#importing the libraries needed for project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score,recall_score,precision_score


#import the dataset using pandas
wine=pd.read_csv(r"C:\Users\GEETANJALI JENA\Downloads\winequality-white.csv",sep=";")
wine

#analyse the dataset and clean the dataset
wine.head()
wine.tail()
wine.info()
wine.describe()
wine.isnull().sum()

#replacing the null values with mean of the thier own feature as all the features are in numerical data
cols=['fixed acidity','volatile acidity','citric acid','chlorides','density','pH','alcohol']
for col in cols:
    wine[col]=wine[col].fillna(wine[col].mean())
wine.isnull().sum()
wine.info()

#histograms of the features of my dataset how well my data is classified 
columns=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
         'total sulfur dioxide','density','pH','sulphates','alcohol','quality']
for col in columns:
    plt.Figure()
    sb.histplot(wine[col],kde=True,bins=15)
    plt.xlabel(col)
    plt.ylabel("frequency")
    plt.title(f"Histogram of {col}")
    plt.show()
    
    
#by seeing the above histograms of each feature, there are some long tails which means a outliers and quality feature is unbalanced so we can convert that it 
#into three categories of qualities like badquality,avgquality and good quality based on some thresholds for bad<=4,average>4 and <=6 and good>6
condition=[wine['quality']<=4,
           (wine['quality']>4) & (wine['quality']<=6),
           wine['quality']>6]
values=['bad','average','good']
wine['Quality']=np.select(condition,values)
wine['Quality'].value_counts()

#countplot
plt.Figure(figsize=(8,4))
sb.countplot(data=wine,x='Quality',palette='Set2')
plt.title("Count plot of Quality")
plt.show()


#check is there any outliers and inliers as per the above histogram by using boxplot
plt.Figure(figsize=(8,20))
sb.boxplot(data=wine.drop(columns=['quality','Quality']))
plt.xticks(rotation=90)
plt.title("Box Plot for outliers and inliers")
plt.show()

#From above we can conclude there are some inliers and outliers so to remove these use IQR Method(Inter Quantile Range)
def remove_outiers(df,cols):
    for col in columns:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1
        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        df[col]=np.where(df[col] > upper, upper,
                           np.where(df[col] < lower, lower, df[col]))
    return df

outrange=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
         'total sulfur dioxide','density','pH','sulphates','alcohol']
wine=remove_outiers(wine,outrange)

#Box Plot after removing outliers and inliers
plt.Figure(figsize=(8,20))
sb.boxplot(data=wine.drop(columns=['quality','Quality']))
plt.xticks(rotation=90)
plt.title("Box Plot after removing outliers and inliers")
plt.show()

#By using heatmap can say that how realted each feature to the Quality feature
plt.Figure(figsize=(8,4))
sb.heatmap(wine.drop('Quality',axis=1).corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Heat map to see relations")
plt.show()

#to train the model take independent and dependent and variables
x=wine.drop(['quality','Quality'],axis=1)
y=wine['Quality']
#as we seen in box plot there is a need to do for normalization
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)

#split the model for train and test using traintestsplit
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#use classification models 

#1.logisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
Logistic_prediction=model.predict(x_test)
print("Confusion matrix of Logistic Regression: \n",confusion_matrix(y_test,Logistic_prediction))
print("Classification report of Logistic Regression: \n",classification_report(y_test,Logistic_prediction))

#2.KNN
#For KNN we need to know the value of k and k value always less than the xtrain length
k=np.sqrt(len(x_train))
print(k)#so we can take k<62
model=KNeighborsClassifier(n_neighbors=55)
model.fit(x_train,y_train)
Knnprediction=model.predict(x_test)
print("Confusion matrix of KNN: \n",confusion_matrix(y_test,Knnprediction))
print("Classification report of KNN: \n",classification_report(y_test,Knnprediction))

#3.SVC
model=SVC(kernel='linear')
model.fit(x_train,y_train)
svcprediction=model.predict(x_test)
print("Confusion matrix: ",confusion_matrix(y_test,svcprediction))
print("Classification report: ",classification_report(y_test,svcprediction))

#4.Decision Tree
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
treeprediction=model.predict(x_test)
print("Confusion matrix: ",confusion_matrix(y_test,treeprediction))
print("Classification report: ",classification_report(y_test,treeprediction))

#5.Random Forest
model=RandomForestClassifier()
model.fit(x_train,y_train)
forestprediction=model.predict(x_test)
print("Confusion matrix: ",confusion_matrix(y_test,forestprediction))
print("Classification report: ",classification_report(y_test,forestprediction))

#Diagrams of the above all models confusion matrices
preds=[Logistic_prediction,Knnprediction,svcprediction,treeprediction,forestprediction]
names=['Logistic Regression','KNN','SVC','Decision Tree','Random Forest']
mapcolor=['Blues','Oranges','Greens','Oranges','Reds']
labels=['bad','average','good']

for i in range(5):
    cm=confusion_matrix(y_test,preds[i],labels=labels)
    plt.Figure(figsize=(5,4))
    sb.heatmap(cm,annot=True,cmap=mapcolor[i],xticklabels=labels,yticklabels=labels)
    plt.title(f"{names[i]}- Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

#Check the cross valdiation score
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()  
scores = cross_val_score(model, x_train, y_train, cv=5)  

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
