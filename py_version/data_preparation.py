import numpy as np
import torch
from torch.utils.data import TensorDataset , DataLoader
import yaml
import os 

# device 路径 batchsize 
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

device = config['device']
batchsize = config['batch_size']
data_dir = config['data_dir']


# to tensor to dataset to dataloader to GPU
x_data_set = np.load(os.path.join(data_dir , 'rand_segment_x_data_set.npy'))
y_data_set = np.load(os.path.join(data_dir , 'rand_segment_y_data_set.npy'))
U = np.load(os.path.join(data_dir , 'u.npy'))
sigma = np.load(os.path.join(data_dir , 'sigma.npy'))
Vt = np.load(os.path.join(data_dir , 'Vt.npy'))



x_train_tensor = torch.tensor(x_data_set[:,:1000],device=device).float().permute(1,0).unsqueeze(1)
x_test_tensor = torch.tensor(x_data_set[:,1000:],device=device).float().permute(1,0).unsqueeze(1)
y_train_tensor = torch.tensor(y_data_set[:,:1000],device=device).float().permute(1,0).unsqueeze(1)
y_test_tensor = torch.tensor(y_data_set[:,1000:],device=device).float().permute(1,0).unsqueeze(1)


train_dataset = TensorDataset(y_train_tensor,x_train_tensor) # exchange the position of x and y because the input should be y
test_dataset = TensorDataset(y_test_tensor,x_test_tensor)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batchsize
)
