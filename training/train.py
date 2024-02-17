#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from options.option import parse_args
from utils import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F
import torch.nn.functional as FN
from validation.validate import *

logger = logging.getLogger()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

LAMBDA = 0.001
LR_DISCR = 0.0001


# Main train loop
#   args:
#       - args: the command line arguments
#       - model: Semantic Segmentation model (Bisenet)
#       - optimizer: the optimizer chosen
#       - dataloader_train: train dataloader
#       - dataloader_val: validation dataloader
#       - start_epoch: starting epoch, > 0 if resuming from another training
#


def train(args, model, optimizer, dataloader_train, dataloader_val,start_epoch, comment=''):
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    # Cross Entropy Loss used
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    max_miou = 0
    step = start_epoch

    # Train loop
    for epoch in range(start_epoch,args.num_epochs):
        
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        # Iterate over the train dataloader
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            # Execute the model and calculate the loss
            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            # Backward
            scaler.scale(loss).backward()
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

        # Perform the validation
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



# Adversarial domain adaptation train loop
#   args:
#       - args: the command line arguments
#       - model: Semantic Segmentation model (Bisenet)
#       - model_D1: Discriminator model
#       - optimizer: the optimizer for the model
#       - optimizer_D1: the optimizer for the disciminator
#       - dataloader_source: source dataloader (GTA5)
#       - dataloader_target: target dataloader (Cityscapes)
#       - dataloader_val: validation dataloader
#       - start_epoch: starting epoch, > 0 if resuming from another training
#

def train_and_adapt(args, model, model_D1, optimizer,optimizer_D1, dataloader_source, dataloader_target, dataloader_val, start_epoch, comment=''):
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()
    
    # Cross Entropy Loss for the model and BCE Loss for the discriminator
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_miou = 0
    step = start_epoch
    source_label = 0
    target_label = 1

    # Train loop
    for epoch in range(start_epoch,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        discr_lr = poly_lr_scheduler(optimizer_D1,args.lr_discr,iter=epoch,max_iter=args.num_epochs)

        model.train()
        model_D1.train()


        tq = tqdm(total=len(dataloader_target) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_discr %f' % (epoch, lr,discr_lr))
        
        loss_record = []
        loss_source_record = []
        loss_target_record = []
        
        # Iterate over the source and target dataset, coupled with the zip function
        for i, ((src_x, src_y), (trg_x, _)) in enumerate(zip(dataloader_source, dataloader_target)):
            trg_x = trg_x.cuda()
            src_x = src_x.cuda()
            src_y = src_y.long().cuda()
            
            optimizer.zero_grad()
            optimizer_D1.zero_grad()

            #Train with source, in this phase don't train the discriminator
            for param in model_D1.parameters():
                param.requires_grad = False

            # Calculate the loss for the Segmentation
            with amp.autocast():
                output_s, out16_s, out32_s = model(src_x)
                loss1 = loss_func(output_s, src_y.squeeze(1))
                loss2 = loss_func(out16_s, src_y.squeeze(1))
                loss3 = loss_func(out32_s, src_y.squeeze(1))
                loss = loss1 + loss2 + loss3

            # Backward
            scaler.scale(loss).backward()

            # Calcolate the loss trying to fool the discriminator, the discriminator should not distinguish
            # between the source and target prevision
            with amp.autocast():
                output_t, out16_t, out32_t = model(trg_x)
                
                D_out1 = model_D1(FN.softmax(output_t,dim=1))

                loss_adv_target1 = bce_loss(D_out1,torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

                loss_f = args.lambda_d1*loss_adv_target1

                loss = loss + loss_f

            # Backward
            scaler.scale(loss_f).backward()
            
            
            loss_record.append(loss.item())

            #Train D, now train also the discriminator
            for param in model_D1.parameters():
                param.requires_grad = True

            output_t = output_t.detach()
            output_s = output_s.detach()

            # Train the discriminato to distinguish from source and target domain prevision
            with amp.autocast():
                D_out1_s = model_D1(FN.softmax(output_s,dim=1))
                loss_d1_s = bce_loss(D_out1_s,torch.FloatTensor(D_out1_s.data.size()).fill_(source_label).cuda())
            
            # Backward
            scaler.scale(loss_d1_s).backward()
            
            with amp.autocast():
                D_out1_t = model_D1(FN.softmax(output_t,dim=1))
                loss_d1_t = bce_loss(D_out1_t,torch.FloatTensor(D_out1_t.data.size()).fill_(target_label).cuda())    
            
            # Backward
            scaler.scale(loss_d1_t).backward()
            
            
            scaler.step(optimizer_D1)
            scaler.step(optimizer)
            scaler.update()


            tq.update(args.batch_size)

            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            
            loss_source_record.append(loss_d1_s.item())
            loss_target_record.append(loss_d1_t.item())

        tq.close()
        
        loss_train_mean = np.mean(loss_record)
        loss_discr_source_mean = np.mean(loss_source_record)
        loss_discr_target_mean = np.mean(loss_target_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        print('loss for discriminator source: %f' % (loss_discr_source_mean))
        print('loss for discriminator target: %f' % (loss_discr_target_mean))

        # Save a checkpoint of the model, the discriminator and the optimizers
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))
            torch.save({'state_dict':model_D1.state_dict(),'optimizer_state_dict': optimizer_D1.state_dict()}, os.path.join(args.save_model_path, 'latest_discr_'+str(epoch)+'.pth'))
        
        # Perform evaluation
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
    val(args, model, dataloader_val, writer, epoch, step)

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

def train_improvements(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, start_epoch, L, comment=''):
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
                
       
            # 1. source to target, target to target
            src_in_trg = FDA_source_to_target( src_img, trg_img, L)             # scr,src_lbl
            trg_in_trg = trg_img.cuda()                                              # trg, trg_lbl
           
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