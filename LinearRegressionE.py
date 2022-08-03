from pickle import NONE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import TrainingData
import TrainModel
import PlotObject

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

def createDataFrame(randomForestModel):
    lr_results = pd.DataFrame({'Linear Regression with e_Mbh': [randomForestModel.get_train_mse(), randomForestModel.get_train_r2(), randomForestModel.get_test_mse(), randomForestModel.get_test_r2()]})
    lr_results.index = ['Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    return lr_results


def main():
    dataset = pd.read_csv('Data/BlackHoles.csv', sep=';')
    dataset = dataset.fillna(0)

    y = dataset['Mbh']
    x = dataset[['GLON', 'GLAT', 'Dist', 'z', 'e_Mbh']]

    trainData = getData(x, y)

    randomForestModel = RandomFor(trainData)

    lr_results = createDataFrame(randomForestModel)

    PLTObject = PlotObject.PLTData()
    PLTObject.set_model(randomForestModel)
    PLTObject.set_trainData(trainData)
    PLTObject.set_lr_results(lr_results)
    return PLTObject


if __name__ == "__main__":
    main()
