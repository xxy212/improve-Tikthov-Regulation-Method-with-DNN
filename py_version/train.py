import numpy as np
import torch
import torch.optim as optim
from data_preparation import train_loader , test_loader 
from models import SimpleCNN , U_Net , ResUNet
from pipeline import StepByStep
import yaml
import os 

# 路径 优化器 训练参数 
with open('config.yaml','r')as file:
    config = yaml.safe_load(file)

data_dir = config['data_dir']
modelname = config['model']
device = config['device']
alpha = config['alpha']
EPOCH = config['epoch']
lr = config['lr']

in_ch = config['in_ch']
out_ch = config['out_ch']
# standard pipeline of model training


# loss function
# def F_alphas(x1,x2,alphas):
#     return (x1)/x2+alphas
sigma = np.load(os.path.join(data_dir ,'sigma.npy'))
U = np.load(os.path.join(data_dir,'u.npy'))
Vt = np.load(os.path.join(data_dir,'Vt.npy'))

# transform the singular system into torch.tensor
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
U = torch.tensor(U, device=device).float()
V = torch.tensor(Vt.T, device=device).float()  # transpose Vt and get V
sigma = torch.tensor(sigma, device=device).float()

def loss_function(output, input, xgt, alpha=alpha):
    """
    
    Args:
        output (torch.Tensor):  (batch_size, 2, n)
        input (torch.Tensor):  (batch_size, n)
        xgt (torch.Tensor):  (batch_size, n)
        alpha (float)

    Returns:
        loss
    """
    batch_size = input.shape[0]
    n = input.shape[2]
    
    # 提取模型输出的两个通道
    psi1 = output[:, 0, :]
    psi2 = output[:, 1, :]
    
    loss = 0.0
    for batch in range(batch_size):
        for k in range(n):
            factor = (sigma[k]**-1) * (psi1[batch, k] / (psi2[batch, k] + alpha))
            in_product_y = torch.matmul( input[batch, 0,:] , U[:, k] )
            in_product_xgt = torch.matmul(xgt[batch,0, :], V[:, k])
            loss += ((factor * in_product_y) - (in_product_xgt))**2

    return loss /(batch_size*n)




# choose model
if modelname == 'SimpleCNN':
    print(f'start to train {modelname}')
    model = SimpleCNN(in_channels=in_ch,out_channels=out_ch)
    
elif modelname == 'Unet':
    print(f'start to train {modelname}')
    model = U_Net(in_ch=in_ch,out_ch=out_ch)
    
elif modelname =='ResUnet':
    print(f'start to train {modelname}')
    model = ResUNet(in_channels=in_ch,out_channels=out_ch)
    

optimizer = optim.Adam(model.parameters(),lr=lr)
pipeline = StepByStep(model=model,loss_fn=loss_function,optimizer=optimizer,alpha=alpha)

pipeline.set_seed(13)

if 'noise' in data_dir:
    pipeline.set_tensorboard(name=f'train_noised_alpha{alpha}')
else:
    pipeline.set_tensorboard(name=f'train_data_{alpha}')
    

pipeline.set_loaders(train_loader=train_loader,val_loader=test_loader)
pipeline.train(EPOCH)

os.makedirs('trained_model',exist_ok=True)

if 'noise' in data_dir:
    pipeline.save_checkpoint(filename=f'trained_model/{modelname}_random_segment_{EPOCH}_epochs_alpha={alpha}_noised.pth')
else:
    pipeline.save_checkpoint(filename=f'trained_model/{modelname}_random_segment_{EPOCH}_epochs_alpha={alpha}.pth')
