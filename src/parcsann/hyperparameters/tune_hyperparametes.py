from parcsann.read_data.nuclear_code_data import CoreData


class TuneHyperparametes:
    def __init__(self, core_data: CoreData):
        self.X_train, self.X_val, self.y_train, self.y_val = core_data.train_test_div()
        