import os
import cv2
import torch
import random
import imageio
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

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
        step = 3
        if (125//n_frames)<step:
            step=1
        length=(n_frames-1)*step
        idxs=[]
        start=np.random.randint(0,125-length)
        for j in range(n_frames):
            idxs.append(start)
            start+=step
    else:
        idxs=list(range(0,125))
    videos=os.listdir(os.path.join(root_path,'X'))
    video_input_path = os.path.join(root_path, 'X')
    dataset = []
    video_target_path = os.path.join(root_path, 'Y')
    for video in videos:
        input_path = os.path.join(video_input_path, video)
        target_path = os.path.join(video_target_path, video.replace('X','Y'))
        sample = { 'video': input_path, 'n_frames': 125,'video_id': video,'target_video': target_path}
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
        return clip,target_clip

    def __len__(self):
        return len(self.data)

def videopadding(imgClip,stride,n_frames):
    b,t,c,h,w=imgClip.size()
    nums=(n_frames//2)*stride
    idxs=list(range(nums-1,-1,-1))
    imgClip_f=imgClip[:,:nums]
    imgClip_f=imgClip_f[:,idxs]
    out=torch.cat([imgClip_f,imgClip],dim=1)

    imgClip_b=imgClip[:,-nums:]
    imgClip_b=imgClip_b[:,idxs]
    out=torch.cat([out,imgClip_b],dim=1)
    return out