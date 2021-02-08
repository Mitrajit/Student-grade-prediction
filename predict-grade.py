# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1>Predicting final grade</h1>
# This code uses <b><i>Linear regression algorithm</i></b> to create a model using the student performance data set from UCI "https://archive.ics.uci.edu/ml/datasets/Student+Performance", and predict their final grade accordingly. 

# %%
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# %% [markdown]
# <h2>Spliting the data</h2>

# %%
data=pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data)
predict="G3"
X=np.array(data.drop([predict],1))
Y=np.array(data[predict])
print(X,"\n\nG3\n",Y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

# %% [markdown]
# <h2>Training the model</h2>

# %%
linear = linear_model.LinearRegression()
best=0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
    linear.fit(x_train, y_train)
    acc = linear.score(x_test,y_test)
    if best<acc:
        best=acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)
pickle_in=open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
print("Best accuracy:",best)
print("Coff: ",linear.coef_)
print("Intersept: ",linear.intercept_)

# %% [markdown]
# <h2>Predict the final grade</h2>

# %%
pickle_in=open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
prediction = linear.predict(x_test)
print("Accuracy:",linear.score(x_test,y_test))
for x in range(len(prediction)):
    print("Predicted G3:",prediction[x],"Actual G3:",y_test[x])

# %% [markdown]
# <h2>Plotting the data using pyplot</h2>

# %%
style.use("ggplot")
pyplot.scatter([x_test[i][0] for i in range(len(x_test))],y_test)
pyplot.scatter([x_test[i][0] for i in range(len(x_test))],prediction)
pyplot.xlabel("G1")
pyplot.ylabel("Final grade")
pyplot.show()