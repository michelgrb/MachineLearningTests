from pickle import NONE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import TrainingData
import TrainModel


def getData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=45)
    
    trainData = TrainingData.Training()
    trainData.set_x_train(x_train)
    trainData.set_x_test(x_test)
    trainData.set_y_train(y_train)
    trainData.set_y_test(y_test)

    return trainData


def RandomFor(trainData):
    lr = RandomForestRegressor(max_depth=7, random_state=45)
    lr.fit(trainData.get_x_train(), trainData.get_y_train())

    y_lr_train_prediction = lr.predict(trainData.get_x_train())
    y_lr_test_prediction = lr.predict(trainData.get_x_test())

    lr_train_mse = mean_squared_error(
        trainData.get_y_train(), y_lr_train_prediction)
    lr_train_r2 = r2_score(trainData.get_y_train(), y_lr_train_prediction)
    lr_test_mse = mean_squared_error(
        trainData.get_y_test(), y_lr_test_prediction)
    lr_test_r2 = r2_score(trainData.get_y_test(), y_lr_test_prediction)
    
    randomForestModel = TrainModel.Train()
    randomForestModel.set_train_prediction(y_lr_train_prediction)
    randomForestModel.set_test_prediction(y_lr_test_prediction)
    randomForestModel.set_train_mse(lr_train_mse)
    randomForestModel.set_train_r2(lr_train_r2)
    randomForestModel.set_test_mse(lr_test_mse)
    randomForestModel.set_test_r2(lr_test_r2)

    return randomForestModel


def NearestNeighbors(trainData):

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(trainData.get_x_train(), trainData.get_y_train())

    y_nb_train_prediction = classifier.predict(trainData.get_x_train())
    y_nb_test_prediction = classifier.predict(trainData.get_x_test())

    nb_train_mse = mean_squared_error(
        trainData.get_y_train(), y_nb_train_prediction)
    nb_train_r2 = r2_score(trainData.get_y_train(), y_nb_train_prediction)
    nb_test_mse = mean_squared_error(
        trainData.get_y_test(), y_nb_test_prediction)
    nb_test_r2 = r2_score(trainData.get_y_test(), y_nb_test_prediction)
    
    nearestNeighborsModel = TrainModel.Train()
    nearestNeighborsModel.set_train_prediction(y_nb_train_prediction)
    nearestNeighborsModel.set_test_prediction(y_nb_test_prediction)
    nearestNeighborsModel.set_train_mse(nb_train_mse)
    nearestNeighborsModel.set_train_r2(nb_train_r2)
    nearestNeighborsModel.set_test_mse(nb_test_mse)
    nearestNeighborsModel.set_test_r2(nb_test_r2)

    return nearestNeighborsModel


def createDataFrame(nearestNeighborsModel, randomForestModel):
    lr_results = pd.DataFrame({'Linear Regression': [randomForestModel.get_train_mse(), randomForestModel.get_train_r2(), randomForestModel.get_test_mse(), randomForestModel.get_test_r2()],
                               'NearestNeighbors': [nearestNeighborsModel.get_train_mse(), nearestNeighborsModel.get_train_r2(), nearestNeighborsModel.get_test_mse(), nearestNeighborsModel.get_test_r2()]})
    lr_results.index = ['Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    return lr_results


def getOutput(lr_results, trainData, nearestNeighborsModel, randomForestModel):
    print(lr_results)

    plt.scatter(x=trainData.get_y_test(),
                y=randomForestModel.get_test_prediction(), alpha=0.3, color='red')
    plt.scatter(x=trainData.get_y_test(),
                y=nearestNeighborsModel.get_test_prediction(), alpha=0.3, color='blue')

    z = np.polyfit(trainData.get_y_test(), randomForestModel.get_test_prediction(), 1)
    p = np.poly1d(z)
    plt.plot(trainData.get_y_train(), p(trainData.get_y_train()), '#DD6868')

    z = np.polyfit(trainData.get_y_test(),
                   nearestNeighborsModel.get_test_prediction(), 1)
    p = np.poly1d(z)
    plt.plot(trainData.get_y_train(), p(trainData.get_y_train()), '#0F275C')

    plt.ylabel('Prediction Age')
    plt.xlabel('Experimental Age')

    plt.show()


def main():
    dataset = pd.read_csv('Data/abalone_train.csv')
    dataset = dataset.fillna(0)

    y = dataset['Age']
    x = dataset.drop(['Age'], axis=1)

    trainData = getData(x, y)

    nearestNeighborsModel = NearestNeighbors(trainData)
    randomForestModel = RandomFor(trainData)

    lr_results = createDataFrame(nearestNeighborsModel, randomForestModel)

    getOutput(lr_results, trainData, nearestNeighborsModel, randomForestModel)


if __name__ == "__main__":
    main()
