import os
import cv2
import imageio
import numpy as np
import torch.nn.functional as F

from PIL import Image
from dataset.dataset import *
from model.network import MaskUNet,generator

video_path='video_decaptioning/test_imgs'
n_frames=125
mask_model_path='mask_extraction/checkpoint/MaskExtractor.pth'
model_G_path='video_decaptioning/chekpoint/netG.pth'
T=7
s=3

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
if __name__ == '__main__':
    masknet = MaskUNet(n_channels=3, n_classes=1)
    masknet.load_state_dict(torch.load(mask_model_path))
    masknet=masknet.cuda()
    masknet=torch.nn.DataParallel(masknet,device_ids=[0])
    masknet.eval()
    net_G=generator()

    net_G.load_state_dict(torch.load(model_G_path))
    net_G=net_G.cuda()
    net_G =torch.nn.DataParallel(net_G, device_ids=[0])    
    net_G.eval()

    frames = np.empty((125, 128, 128, 3), dtype=np.float32)
    for i in range(125):
        img_file = os.path.join(video_path,'{:03d}.png'.format(i+1))
        raw_frame = np.array(Image.open(img_file).convert('RGB'))/255
        frames[i] = raw_frame
    frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float().cuda()
    frames=(frames-0.5)/0.5
    frames=frames.unsqueeze(0)
    with torch.no_grad():
        masks=masknet(frames)
        masks = (masks > 0.5).float().cuda()
        frames_padding=videopadding(frames,s,T).cuda()  
        masks_padding=videopadding(masks,s,T).cuda()   
        pred_imgs=[]
        for j in range(125):
            input_imgs=frames_padding[:,j:j+(T-1)*s+1:s]
            input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
            pred_img= net_G(input_imgs,input_masks)
            pred=transforms.ToPILImage()(pred_img.squeeze(0)*0.5+0.5).convert('RGB')
            pred.save('video_decaptioning/test_imgs/%03d.png'%(j))
            mask=masks[:,j].squeeze(0).permute(1,2,0).cpu().numpy()*255
            cv2.imwrite('video_decaptioning/test_imgs/%03d.png'%(j),mask)
            pred_imgs.append(pred_img*0.5+0.5)
        video=torch.cat(pred_imgs,dim=0)
        video=(video.cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)
        imageio.mimwrite(os.path.join('video_decaptioning/test_imgs','video.mp4'),video,fps=25,quality=8,macro_block_size=1)