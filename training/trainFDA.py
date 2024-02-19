#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from options.option import parse_args
from utils.utils import *
from utils.FDAUtils import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F
import torch.nn.functional as FN
from validation.validate import *

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

# Main train loop
#   args:
#       - args: the command line arguments
#       - model: Semantic Segmentation model (Bisenet)
#       - optimizer: the optimizer chosen
#       - dataloader_source: source dataloader (GTA5)
#       - dataloader_target: train dataloader (Cityscapes)
#       - dataloader_val: validation dataloader (cityscapes)
#       - L: L for fourier transform
#       - start_epoch: starting epoch, > 0 if resuming from another training
#

def train_FDA(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, start_epoch, L, comment=''):
    args = parse_args()
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()
    # Cross Entropy Loss on source combined with EntLoss on target (a function that aims to lower the entropy)
    loss_crossE = torch.nn.CrossEntropyLoss(ignore_index=255) 
    loss_ent = EntLoss()

    max_miou = 0
    step = start_epoch
    
    # Train loop
    for epoch in range(start_epoch,args.num_epochs):
        
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        model.train()
        
        tq = tqdm(total=len(dataloader_target) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        
        loss_record = []
        
        # Iterate over the source and target dataset, coupled with the zip function
        for i, ((src_img, src_lbl), (trg_img, trg_lbl)) in enumerate(zip(dataloader_source, dataloader_target)):
           
            mean_img = torch.zeros(1,1)
            if mean_img.shape[-1] < 2:
                B, C, H, W = src_img.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
                
       
            # source to target, target to target
            #function for the Fourier Transform
            src_in_trg = FDA_source_to_target( src_img, trg_img, L)             
            trg_in_trg = trg_img.cuda()                                         
           
            optimizer.zero_grad()
            
            src_img= src_in_trg.cuda()
            src_lbl= src_lbl.long().cuda()
            trg_img= trg_in_trg.cuda()
            trg_lbl= trg_lbl.long().cuda()

            #Train with source
            with amp.autocast():
                output_s, out16_s, out32_s = model(src_img.half())
                loss1 = loss_crossE(output_s, src_lbl.squeeze(1))
                loss2 = loss_crossE(out16_s, src_lbl.squeeze(1))
                loss3 = loss_crossE(out32_s, src_lbl.squeeze(1))
                loss = loss1 + loss2 + loss3

            #Train with target
            with amp.autocast():
                output_t, out16_t, out32_t = model(trg_img)
                lossT = loss_ent(output_t, args.ita)
            
            # Use the EntLoss function only if a certain epoch is reached 
            triger_ent = 0.0
            if epoch > args.switch2entropy:
                triger_ent = 1.0

            # Compute the loss
            loss=loss+triger_ent*lossT*args.entW

            # Backward
            scaler.scale(loss).backward()

            loss_record.append(loss.item())
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        # Save a checkpoint of the model and the optimizer
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))

        # Perform the evaluation
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, writer, epoch, step)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    
    #final evaluation
    precision, miou = val(args, model, dataloader_val, writer, epoch, step)
    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou val', miou, epoch)