import os
import cv2
import torch
import random
import imageio
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

def blur(mask,kernel_size,threshold):
    mask_clip=torch.mean(mask, dim=1, keepdim=True)
    T,c,h,w=mask_clip.size()
    mask_array=mask_clip.permute([0,2,3,1]).numpy()
    result=[]
    for i in range(T):
        mask=mask_array[i]
        out=cv2.blur(mask,kernel_size)
        out=(out>threshold).astype(np.float)
        result.append(out)
    result=torch.Tensor(result).unsqueeze(1)
    return result

def video_loader(path,target_path,frame_indices,training):
    inputs,targets = [],[]
    if training:
        p=random.randint(0,1)
        transfrom=transforms.Compose([transforms.RandomHorizontalFlip(p),#随机水平反转，增强数据,
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    else:
        transfrom=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    for i in frame_indices:
        image_path = os.path.join(path, '{:03d}.png'.format(i+1))
        t_path=os.path.join(target_path, '{:03d}.png'.format(i+1))
        img=Image.open(image_path)
        target=Image.open(t_path)
        img_tensor=transfrom(img).unsqueeze(0)
        target_tensor=transfrom(target).unsqueeze(0)
        inputs.append(img_tensor)
        targets.append(target_tensor)
    inputs=torch.cat(inputs,dim=0) #size[n,3,128,128]
    targets=torch.cat(targets,dim=0)
    return inputs,targets

def make_dataset(root_path,n_frames,training):
    if training:
        step = 1
        if (125//n_frames)<step:
            step=1
        length=(n_frames-1)*step
        idxs=[]
        start=np.random.randint(0,125-length)
        for j in range(n_frames):
            idxs.append(start)
            start+=step
        N=70000
    else:
        idxs=list(range(0,125))
        N=5000
    video_input_path = os.path.join(root_path, 'X')
    dataset = []
    video_target_path = os.path.join(root_path, 'Y')
    for n in range(N):
        video_name='{:03d}'.format(n)
        input_path = os.path.join(video_input_path, video_name)
        target_path = os.path.join(video_target_path, video_name)
        sample = { 'video': input_path, 'n_frames': 125,'video_id': video_name,'target_video': target_path}
        sample['frame_indices'] = idxs
        dataset.append(sample)
    return dataset

class VideoDecaptionData(data.Dataset):
    def __init__(self,root_path,n_frames,training=True):
        self.n_frames=n_frames
        self.data = make_dataset(root_path,n_frames,training)
        self.training=training

    def __getitem__(self, index):
        path = self.data[index]['video']
        target_path = self.data[index]['target_video']
        frame_indices = self.data[index]['frame_indices']
        clip,target_clip = video_loader(path,target_path, frame_indices,self.training)
        mask=torch.abs(clip-target_clip)
        mask_clip = blur(mask,(5,5),0.05)
        return clip,mask_clip

    def __len__(self):
        return len(self.data)