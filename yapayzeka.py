#Yapay Zeka Ödevi Serkan Baykal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

#Dosya Yolu
cwd = os.getcwd()

#Veri Yolu
readCsv = cwd + "/data.csv"

#Verileri Oku
df = pd.read_csv(readCsv)

#İlk 5 Satır
df.head()

#1- Veri setinde hastalıklı ve sağlam sayılarını sütun grafiği kullanarak çizdirin.
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))

#2- Cinsiyete göre hasta ve sağlıklı hasta sayıları sütun grafiği ile ifade ediniz. 
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['red','green' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0 = Female, 1 = Male')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#3-Veri setindekilerin yaş dağılımını gösteren bir sütun grafiği çizdirin.
sns.countplot(x='age', data=df, palette="mako_r")
plt.xlabel("Age")
plt.show()

#4-Veri setinde hasta olanların yaş dağılımını gösteren bir sütun grafiği çizdirin.
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()

df = df.drop(columns = ['cp', 'thal', 'slope'])

y = df.target.values
x_data = df.drop(['target'], axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#Matris
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


#Oluşturma
def initialize(dimension):
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias


def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head

def forwardBackward(weight,bias,x_train,y_train):
    #Forward
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    #Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    return cost,gradients

def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    #Verileri Güncelle
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        costList.append(cost)
        index.append(i)
    parameters = {"weight": weight,"bias": bias}
    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients

def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)
    y_prediction = np.zeros((1,x_test.shape[1]))
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction

#7-Logistic Regresyon kullanarak veri setini sınıflandırın 
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
logistic_regression(x_train,y_train,x_test,y_test,1,100)
accuracies = {}
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))

# 8- K-NN kullanarak veri setini sınıflandırın 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("KNN Score: {:.2f}%".format(knn.score(x_test.T, y_test.T)*100))

#Find "k" Value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracies['KNN'] = acc
print("Maximum KNN Score: {:.2f}%".format(acc))

#9- Naive Bayes kullanarak veri setini sınıflandırın 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

#10- Karar ağaçları kullanarak veri setini sınıflandırın 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))

from sklearn.metrics import confusion_matrix

#Keras Kullanımı İçin, Tensorflow Kütüphanesi Gerekli
from keras.models import Sequential
from keras.layers import Dense

#Modelleri Karşılaştırma
colors = ["red", "green", "orange", "magenta","turquoise","black"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

#Karmaşıklık Matrisi
y_head_lr = lr.predict(x_test.T)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train.T, y_train.T)
y_head_knn = knn3.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)


cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)

cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()

#11- Yapay Sinir Ağları kullanarak veri setini sınıflandırın 
plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(),robust=True,annot=True)

#Karışık Verileri Tespit Etme
plt.subplots(figsize=(15,6))
df.boxplot(patch_artist=True, sym="k.")
X = df.iloc[:, 1:11].values
y = df.iloc[:, -2].values

#Ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
#X = np.array(X).reshape(-1,1)
X = sc_X.fit_transform(X)

#y = np.array(y).reshape(-1,1)
y = sc_y.fit_transform(y)

#Eğitim ve Test Setleri Oluşturma
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import Imputer

imputer= Imputer(strategy='mean')
imputer = imputer.fit(X[:,4:11])
X[:,4:11]= imputer.transform(X[:,4:11])

#Yapay Sinir Ağı
classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fit
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)























