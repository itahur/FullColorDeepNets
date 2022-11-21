import torch

class colorMNist(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        return self.data[id][0], self.data[id][1]