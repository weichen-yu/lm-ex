import torch
import torch.nn.functional as F

a =  torch.tensor(range(25)).reshape(1,1,5,5)
k1 =  torch.tensor(range(9)).reshape(1,1,3,3)
k2 =  torch.tensor(range(1,10)).reshape(1,1,3,3)

r1 = F.conv2d(a,k1,padding=2)#[:,:,4:-4,4:-4]
r2 = F.conv2d(r1,k2,padding=2)#[:,:,4:-4,4:-4]

k1 = torch.flip(k1,(2,3))#.repeat(1,1,3,3)
r3 = F.conv2d(k1,k2,padding=2)#[:,:,2:-2,2:-2]
r3 = torch.flip(r3,(2,3))#.repeat(1,1,3,3)
r4 = F.conv2d(a,r3,padding=4)#[:,:,4:-4,4:-4]

print(r2.shape,r2)
print(r4.shape,r4)