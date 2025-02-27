# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

class DatasetMemory:

    def __init__(self, dataset):
        self.dataset_original = dataset
        self.dataset = dataset
        self.history = []

    def update_dataset(self, new_dataset, operation):
        self.dataset = new_dataset
        self.history.append(operation)

    def get_history(self):
        return "\n".join(self.history)