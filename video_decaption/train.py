import os
import sys
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from dataset.dataset import *
from utils.cal_ssim import ssim
from model.network import MaskUNet,generator
from torchvision import models
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def eval_net(net,mask_net,writer,T,epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_net.eval()
    mse=0
    val_data = VideoDecaptionData(args.v_root,n_frames,False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    for step, (inputs,targets) in enumerate(val_loader):
        with torch.no_grad():
            masks=mask_net(inputs.cuda())
            masks = (masks > 0.5).float().cuda()
            input_padding=videopadding(inputs,s,T).cuda()  
            masks_padding=videopadding(masks,s,T).cuda()  
            pred_imgs=[]
            criterion = nn.MSELoss()
            s=3
            for j in range(125):
                input_imgs=input_padding[:,j:j+(T-1)*s+1:s]
                input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
                pred_img= net(input_imgs,input_masks)
                pred_imgs.append(pred_img)
            pred_imgs=torch.cat(pred_imgs,dim=0)*0.5+0.5
            t_imgs=targets.squeeze(0)*0.5+0.5
            mse+=criterion(pred_imgs,t_imgs.cuda())
        if step==299:
            writer.add_scalar('Valid_MSE', mse.item(), epoch+1)
            break
    return mse.item()

def crit_ssim(pred,true):
    pred=pred*0.5+0.5
    true=true*0.5+0.5
    return (1-ssim(pred,true))/2

#训练前参数的设定
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_root',type=str, default='train/img',
                        help='Trainset root')
    parser.add_argument('--v_root',type=str, default='dev/img',
                        help='Validationset root')     
    parser.add_argument('--m_root',type=str, default='video_decaptioning/checkpoint',
                        help='Model and Optim saved root')
    parser.add_argument('--mask_model',type=str, default='mask_extraction/checkpoint/MaskExtractor.pth',
                        help='Mask extraction')
    parser.add_argument('--model_path',type=str, default='video_decaptioning/checkpoint/netG.pth',
                        help='Model saved path')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='N_frames in each video')
    parser.add_argument('--T',type=int,default=5,
                        help='T frames in each batch')                                
    parser.add_argument('--epochs',type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', type=int,default=64,
                        help='Batch size')
    parser.add_argument('--lr',type=float,default=0.0001,
                        help='Learning rate')
    parser.add_argument('--load',type=bool,default=False,
                        help='Load trained model')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    net_G = generator()
    print(net_G)
    if args.load:
        logging.info(f'Model loaded from {args.model_path}')
        net_G.load_state_dict(torch.load(args.model_path))
    net_G=net_G.cuda()
    net_G = torch.nn.DataParallel(net_G, device_ids=[0,1])
    T=args.T
    n_frames=args.n_frames
    
    #加载mask生成模型
    mask_net=MaskUNet(3,1)
    logging.info(f'Load mask extractor model from {args.mask_model}')
    mask_net.load_state_dict(torch.load(args.mask_model))
    mask_net=mask_net.cuda()
    mask_net=torch.nn.DataParallel(mask_net,device_ids=[0,1])
    mask_net.eval() 
    try:
        train_data = VideoDecaptionData(args.t_root, n_frames)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        n_train = len(train_loader)
        logging.info(f'''Starting training:
                Epochs:          {args.epochs}
                Batch size:      {args.batch_size}
                Learning rate:   {args.lr}
                Training size:   {n_train}
                Load Model:      {args.load}
            ''')

        optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr,betas=(0.9,0.999))
        criterion1 = nn.L1Loss()
        writer = SummaryWriter(comment=f'Video_Decaptioning')
        valid_mse,valid_n=[],0
        for epoch in range(args.epochs):
            net_G.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
                for global_step, (inputs,targets) in enumerate(train_loader):
                    with torch.no_grad():
                        masks=mask_net(inputs.cuda())
                        masks = (masks > 0.5).float()
                    input_imgs=inputs.cuda()
                    input_masks=masks.cuda()
                    true_img=targets[:,T//2].cuda()
                    mask=input_masks[:,T//2]
                    pred_img=net_G(input_imgs,input_masks)
                    net_G.zero_grad()

                    loss_hole=criterion1(pred_img*mask,true_img*mask)/torch.mean(mask)
                    loss_valid=criterion1(pred_img*(1-mask),true_img*(1-mask))/torch.mean(1-mask)

                    loss_img=6*loss_hole+loss_valid
                    writer.add_scalar('Loss_L1', loss_img.item(), global_step+epoch*n_train)

                    loss_ssim=crit_ssim(pred_img,true_img)
                    writer.add_scalar('Loss_Dssim', loss_ssim.item(), global_step+epoch*n_train)
                    
                    loss_gen=loss_img+loss_ssim
                    loss_gen.backward()
                    optimizer_G.step()
                    pbar.update(1)
            mse=eval_net(net_G,mask_net,writer,T,epoch)
            valid_mse.append(mse)
            if mse==min(valid_mse):
                valid_n=0
                torch.save(net_G.module.state_dict(),os.path.join(args.m_root,'netG.pth'))
                logging.info(f'model {epoch+1} saved!')
            else:
                valid_n+=1
            if valid_n>29:
                logging.info(f'Early Stopping!')
                break        
        writer.close()
    except KeyboardInterrupt:
        torch.save(net_G.module.state_dict(),os.path.join(args.m_root,'netG.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)