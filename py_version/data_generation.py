import numpy as np
import os
import yaml

##  maybe 文件路径
with open ('config.yaml','r') as file:
    config = yaml.safe_load(file)

data_dir = config['data_dir']
os.makedirs(data_dir ,exist_ok=True)

def green_kernel(t,s):
    if t<=s:
        return t*(s-1)
    else:
        return s*(t-1)
a = 0  
b = 1
n = 100
h = (b-a)/n
t_i = np.array([(i-0.5)*(1/n) for i in range(1,n+1)],dtype=float)
s_j = np.array([a + (j-0.5)*h for j in range(1,n+1)],dtype=float)
K = np.zeros((100,100),dtype=float)
for i in range(0,100):
    for j in range(0,100):
        K[i,j] = green_kernel(t_i[i],s_j[j])


U , Sigma ,Vt = np.linalg.svd(K)
np.save(os.path.join(data_dir , 'u.npy') ,U)
np.save(os.path.join(data_dir , 'sigma.npy'),Sigma)
np.save(os.path.join(data_dir , 'Vt.npy'),Vt)


data_set = np.zeros((100,1200))
np.random.seed(17)


for j in range (1200):
    random_segment = 5 + round(np.random.rand()*10)  # random number [5,15] 
    for i in range(0,100,random_segment):
        if i+random_segment<=90:
            data_set[i:i+random_segment,j]=np.random.rand()
        else:
            data_set[i:,j]=np.random.rand()

y_data_set = np.zeros((100,1200))
for i in range(1200):
    y_data_set[:,i] = (K @ data_set[: , i])



np.save(os.path.join(data_dir,'rand_segment_x_data_set.npy'), data_set)

if 'noise' in data_dir:
    noised_y_data_set = np.zeros((100,1200))
    for j in range(1200):
        norm_y = np.linalg.norm(y_data_set[:,j])
        noise = np.random.randn(100)
        norm_noise = np.linalg.norm(noise)
        noised_y_data_set[:,j] = y_data_set[:,j]+0.01*norm_y*noise/norm_noise
    np.save(os.path.join(data_dir,'rand_segment_y_data_set.npy'), noised_y_data_set)
    
else:
    np.save(os.path.join(data_dir,'rand_segment_y_data_set.npy'), y_data_set)
