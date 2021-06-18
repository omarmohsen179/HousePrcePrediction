import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
from sklearn import tree


# import the data
def createModel():
    houseDate = pd.read_csv('train.csv')
    # clean data to features x and label y


    data = houseDate[['2ndFlrSF',
                       '1stFlrSF',
                       'BsmtFullBath',
                       'BsmtHalfBath',
                       'FullBath',
                       'HalfBath',
                       'BedroomAbvGr',
                       'KitchenQual', 'SaleCondition']]

    salesString = data['SaleCondition'].unique()
    kitchenString = data['KitchenQual'].unique()

    for i in range(0, len(data)):
        for j in range(0, len(salesString)):
            if data['SaleCondition'][i] == salesString[j]:
                data['SaleCondition'][i] = j
        for j in range(0, len(kitchenString)):
            if data['KitchenQual'][i] == kitchenString[j]:
                data['KitchenQual'][i] =j
    x = data
    y = houseDate['SalePrice']
    # splite data to train and test much we test much less accuracy
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # create model
    global model
    model = DecisionTreeClassifier()
    # train model
    model.fit(X_train, y_train)
    # create dot file to understand how model workes

    # make prediction
    prediction = model.predict(X_test)
    # y_test = expected values
    # prediction= acuale model values
    joblib.dump(model, 'HousePricePrediction.joblib')
    accuracy = accuracy_score(y_test, prediction)
    print("model accuracy : ", accuracy)


createModel()
model = joblib.load('HousePricePrediction.joblib')
predictionEx = model.predict([[854,856, 1,0,2,1,3,0,0], [854,856, 1,0,2,1,3,0,0]])
print(predictionEx)