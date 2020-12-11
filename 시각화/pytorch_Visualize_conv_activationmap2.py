#https://deepmi.me/2019/02/pytorch-hook-feature-map-%EC%B6%9C%EB%A0%A5%ED%95%98%EA%B8%B0/

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import math

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform=torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize])

def image_url(url):
    img = Image.open(url)
    return img

img=image_url('./dog_sample.png').convert('RGB')
input=transform(img).view(-1,3,224,224)
temp=dict()

# 학습된 vgg19모델을 불러온다
model=torchvision.models.vgg19(True)

# feature_map을 hoop할 함수를 정의한다.
def get_features_hook(self, input, output):
    global count
    temp[count]=torch.squeeze(output).numpy()
    count+=1

for i, _ in model.named_children():
    for k in range(len(model._modules.get(i))):
        model._modules.get(i)[k].register_forward_hook(get_features_hook)
    break # only features

count=0

model.eval()
with torch.no_grad():
    output = model(input)

for i in range(len(temp)):
    fig = plt.figure(figsize=(25, 25), dpi=300)
    for j in range(0, temp[i].shape[0]):
        plt.subplot(int(math.ceil(math.sqrt(temp[i].shape[0]))),int(math.ceil(math.sqrt(temp[i].shape[0]))), j+1)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        plt.imshow(temp[i][j])
        # plt.title(str(j), fontsize=3)
    # fig.suptitle(str(i+1)+"'s feature'", fontsize=3)
    plt.show()
