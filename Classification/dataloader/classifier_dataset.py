import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class signaldataset(Dataset):
    def __init__(self,train, transform = False):
        """
        Initialization method for the dataset.
        
        Parameters:
        - train: training model or testing model.
        """
        self.file_dir = "/home/enigma/3_Liwen/Chatter-Data/Surfaces/clamped"
        self.idx_dir = "/home/enigma/3_Liwen/chatter/UnsupervisedCluster/vgg/pre-process_notebook"
        self.transform = transform
        # self.data_labels = self._load_data_labels()
        if train:
            self.split_index = torch.load(os.path.join(self.idx_dir, "train_index.pt"))
        else:
            self.split_index = torch.load(os.path.join(self.idx_dir, "val_index.pt"))
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # return len(self.data_labels)
        return len(self.split_index)
    
    def __getitem__(self, raw_index):
        """
        Generates one sample of data.
        
        Parameters:
        - index: The index of the sample to return.
        
        Returns:
        A tuple (sample, label) where sample is the data at the given index
        and label is the corresponding label.
        """
        # optical_path, height_path = self.data_labels[index] #filename, optical_path, height_path = self.data_info[index]
        index = self.split_index[raw_index]
        # Load the tensors
        combined_tensor = torch.load(os.path.join(self.file_dir, str(index.item()) + ".pt"))
        if self.transform:
            # Random rotation
            angle = torch.rand(1).item() * 360  # Random angle from 0 to 360 degrees
            combined_tensor = TF.rotate(combined_tensor, angle)
        
        return TF.resize(combined_tensor, [224, 224]), index  # Returning filename instead of index
