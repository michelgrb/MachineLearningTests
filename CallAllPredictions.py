from tempfile import tempdir
import RanddomForest
import LinearRegression
import RanddomForestE
import matplotlib.pyplot as plt
import numpy as np
import LinearRegressionE

def makeplt(temptrainData, tempdataset, tempmodel, x, name):
    plt.subplot(2, 2, x)
    plt.scatter(x=temptrainData.get_y_test(), y=tempmodel.get_test_prediction(), alpha=0.3, color='red')
    z = np.polyfit(temptrainData.get_y_test(), tempmodel.get_test_prediction(), 1)
    p = np.poly1d(z)
    plt.plot(temptrainData.get_y_train(), p(temptrainData.get_y_train()), '#DD6868')
    plt.title('2')
    plt.ylabel('Prediction Mass')
    plt.xlabel('Experimental Mass')
    plt.title(name)
    print(tempdataset)

def main():
    plt.figure() 
    
    temptrainData = LinearRegression.main().get_trainData()
    tempmodel = LinearRegression.main().get_model()
    tempdataset = LinearRegression.main().get_lr_results()
    makeplt(temptrainData, tempdataset, tempmodel, 1, 'LinearRegression without e_Mbh')
    
    temptrainData = LinearRegressionE.main().get_trainData()
    tempmodel = LinearRegressionE.main().get_model()
    tempdataset = LinearRegressionE.main().get_lr_results()
    makeplt(temptrainData, tempdataset, tempmodel, 2, 'LinearRegression with e_Mbh')
    
    temptrainData = RanddomForest.main().get_trainData()
    tempmodel = RanddomForest.main().get_model()
    tempdataset = RanddomForest.main().get_lr_results()
    makeplt(temptrainData, tempdataset, tempmodel, 3, 'RanddomForest without e_Mbh')
    
    temptrainData = RanddomForestE.main().get_trainData()
    tempmodel = RanddomForestE.main().get_model()
    tempdataset = RanddomForestE.main().get_lr_results()
    makeplt(temptrainData, tempdataset, tempmodel, 4, 'RanddomForest with e_Mbh')
    
    plt.show()

if __name__ == "__main__":
    main()
