class PLTData():
    def __init__(self):
        self.lr_results = None
        self.trainData = None
        self.model = None

    # getter methods
    def get_lr_results(self):
        return self.lr_results

    def get_trainData(self):
        return self.trainData

    def get_model(self):
        return self.model

     # setter methods
    def set_model(self, x):
        self.model = x

    def set_trainData(self, x):
        self.trainData = x

    def set_lr_results(self, x):
        self.lr_results = x


