import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from data_preparation import train_dataset , test_dataset
from pipeline import StepByStep
from models import SimpleCNN,ResUNet,U_Net
import os
import torch 
import torch.optim as optim



with open('config.yaml','r')as file:
    config = yaml.safe_load(file)

modelname = config['model']
data_dir = config['data_dir']
in_ch = config['in_ch']
out_ch = config['out_ch']
alpha = config['alpha']
device = config['device']
lr = config['lr']



def reconstruct(output,y,alpha):
    output = output.detach().cpu().numpy()
    inverse_y = np.zeros((1,100))
    sigma = np.load(os.path.join(data_dir ,'sigma.npy'))
    U = np.load(os.path.join(data_dir,'u.npy'))
    Vt = np.load(os.path.join(data_dir,'Vt.npy'))
    
    for k in range(len(sigma)):
        if abs(sigma[k])>1e-3:
            factor = (sigma[k]**-1)*(output[0,k] / (output[1,k] + alpha)) 
            inner_product = np.dot(y[:], U[:, k])
            inner_product = inner_product[0,0]

            inverse_y += factor * inner_product * Vt[k,:]
    return inverse_y

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

if modelname == 'SimpleCNN':
    print(f'start to draw {modelname}')
    model = SimpleCNN(in_channels=in_ch,out_channels=out_ch)
    
elif modelname == 'Unet':
    print(f'start to draw {modelname}')
    model = U_Net(in_ch=in_ch,out_ch=out_ch)
    
elif modelname =='ResUnet':
    print(f'start to draw {modelname}')
    model = ResUNet(in_channels=in_ch,out_channels=out_ch)

optimizer = optim.Adam(model.parameters(),lr=lr)
pipeline = StepByStep(model=model,loss_fn=loss_function,optimizer=optimizer,alpha=1)

if 'noise' in data_dir:
    pipeline.load_checkpoint('trained_model/SimpleCNN_random_segment_100_epochs_noised.pth')
else:
    pipeline.load_checkpoint('trained_model/SimpleCNN_random_segment_100_epochs.pth')

model.eval()

tempt_x ,tempt_y= test_dataset[0]

tempt_x = tempt_x.unsqueeze(0)
output = model(tempt_x).squeeze(0)

tempt_x = tempt_x.detach().cpu()
tempt_y =tempt_y.squeeze().detach().cpu()
inverse_x = reconstruct(output,tempt_x,alpha=0.001)
inverse_x = inverse_x.squeeze()


x_axit = np.linspace(0,100,100)
plt.figure(figsize=(8,6))
plt.plot(x_axit,inverse_x,label = 'reconstruct',linestyle = '--',lw = 2,color = 'b')
plt.plot(x_axit,tempt_y,label = 'gt',linestyle = '-',lw = 1,color = 'r')
plt.legend()
plt.show()


def T_regulation(y ,alpha):
    inverse_y = np.zeros((1,100))
    sigma = np.load(os.path.join(data_dir ,'sigma.npy'))
    U = np.load(os.path.join(data_dir,'u.npy'))
    Vt = np.load(os.path.join(data_dir,'Vt.npy'))
    for k in range(len(sigma)):
        if sigma[k] != 0:
            factor = (sigma[k]**2 / (sigma[k]**2 +alpha))  * (1 / sigma[k])
            inner_product = np.dot(y, U[:, k])
            inverse_y += factor * inner_product * Vt[k,:]
    
    return inverse_y

T_inverse = T_regulation(tempt_x,alpha=0.001).T
x_axit = np.linspace(0,100,100)


plt.plot(x_axit,T_inverse)

tempt_y = tempt_y.squeeze(0).detach().cpu().numpy()
plt.plot(x_axit,tempt_y,linestyle = '--')
plt.show()