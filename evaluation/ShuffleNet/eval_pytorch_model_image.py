import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
#from network import ShuffleNetV1
from network import ShuffleNetV2
import time

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形>转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])


# 加载模型
print('load model begin!')
#model = ShuffleNetV1(group=3)
model = ShuffleNetV2()
model = torch.nn.DataParallel(model)
#checkpoint = torch.load('./2.0x.pth.tar')
checkpoint = torch.load('./ShuffleNetV2.1.5x.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')


# 测试单张图像
img = Image.open('/home/sz/model_eval/panda.jpg')
img = data_transform(img)
#img = torch.Tensor(1,3,224,224) #如果只是测试时间，直接初始化一个Tensor即可
print(type(img))
print(img.shape)
img_f = torch.Tensor(1, 3, 224, 224)
img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
print(type(img))
print(img.shape)

time_c = 0
for i in range(5000):
    print(i)
    time_start = time.time()
    img_= img_f.to(device)
    outputs = model(img_)
    time_end = time.time()
    time_tmp = time_end - time_start
    time_c += time_tmp
#_, predicted = torch.max(outputs,1)
#print('this picture maybe:' + str(predicted))
print('time cost:', time_c, 's')

'''
# 批量测试验证集中的图像，使用dataloader，可以更改batch_size调节测试速度
print('Test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
print(type(test_data))
print('Test data load done!')

torch.no_grad()
for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    print(pred)
    print(label1)
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
'''

