import torch
import configparser


class Evaluation:
    def __init__(self, inputs, denorm_outputs, outputs):
        self.inputs = inputs
        self.denorm_outputs = denorm_outputs
        self.outputs = outputs

        config = configparser.ConfigParser()
        config.read('config.ini')

        

    def mean_deviation():
        pass

    def std_deviation():
        pass

    def mean_absolute_error(self):
        absolute_errors = torch.abs(self.inputs - self.denorm_outputs)
        print(f"Mean absolute error: {torch.mean(absolute_errors)}")

    def mean_squared_error(self):
        pass


