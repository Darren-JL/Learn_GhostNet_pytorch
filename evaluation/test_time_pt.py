import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from ghost_net import ghost_net
import time

device = torch.device('cuda')

print('load model begin!')
model = ghost_net(width_mult=1.0)
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')

torch.no_grad()
#img = Image.open('/home/momo/sun.zheng/GhostNet_tensorflow/source/panda.jpg')
#img = data_transform(img)
time_c = 0
img = torch.Tensor(1,3,224,224)
time_start = time.time()
for i in range(5000):
    print(i)
    img_= img.to(device)
    outputs = model(img_)
time_end = time.time()
time_c = time_end - time_start
#_, predicted = torch.max(outputs,1)
# print('this picture maybe:' + str(predicted))
print('time cost:', time_c, 's')

