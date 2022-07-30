class Train():
    def __init__(self):
        self.train_mse = None
        self.train_r2 = None
        self.test_mse = None
        self.test_r2 = None
        self.nb_test_prediction = None
        self.nb_train_prediction = None

    # getter methods
    def get_train_mse(self):
        return self.train_mse

    def get_train_r2(self):
        return self.train_r2

    def get_test_mse(self):
        return self.test_mse

    def get_test_r2(self):
        return self.test_r2

    def get_train_prediction(self):
        return self.train_prediction

    def get_test_prediction(self):
        return self.test_prediction

    # setter methods
    def set_train_mse(self, x):
        self.train_mse = x

    def set_train_r2(self, x):
        self.train_r2 = x

    def set_test_mse(self, x):
        self.test_mse = x

    def set_test_r2(self, x):
        self.test_r2 = x

    def set_train_prediction(self, x):
        self.train_prediction = x

    def set_test_prediction(self, x):
        self.test_prediction = x

