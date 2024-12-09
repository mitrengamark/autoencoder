class NormalizeData:
    def __init__(self, data):
        self.data = data

    def normalize(self):
        data_min = self.data.min().min()
        data_max = self.data.max().max()
        data_normalized = (self.data - data_min) / (data_max - data_min)

        return data_normalized