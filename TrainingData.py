class Training():
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    # getter methods
    def get_x_train(self):
        return self.x_train

    def get_x_test(self):
        return self.x_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

     # setter methods
    def set_x_train(self, x):
        self.x_train = x

    def set_x_test(self, x):
        self.x_test = x

    def set_y_train(self, x):
        self.y_train = x

    def set_y_test(self, x):
        self.y_test = x
