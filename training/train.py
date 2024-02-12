#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from model.discriminator import FCDiscriminator
from dataset.cityscapes import CityScapes
from dataset.GTA5 import GTA5
import torchvision.transforms as transforms
from torchvision.transforms import v2
from utils import ExtCompose, ExtResize, ExtToTensor, ExtTransforms, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop, ExtColorJitter, FDA_source_to_target
import torch
from torch.utils.data import DataLoader, Subset
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import random
import os
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.autograd import Variable
from validation.validate import *

logger = logging.getLogger()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

LAMBDA = 0.001
LR_DISCR = 0.0001

def train(args, model, optimizer, dataloader_train, dataloader_val,start_epoch, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    max_miou = 0
    step = start_epoch
    for epoch in range(start_epoch,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        #image_number = random.randint(0, len(dataloader_train) - 1)
        for i, (data, label) in enumerate(dataloader_train):
           # colorized_predictions , colorized_labels = GTA5.visualize_prediction(model(data)[0], label)
                
                

            #writer.add_image('eval%d/iter%d/immagineVera' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')
            #writer.add_image('eval%d/iter%d/etichetta' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
           
            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

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
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))

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


def train_and_adapt(args, model, model_D1, optimizer,optimizer_D1, dataloader_source, dataloader_target, dataloader_val, start_epoch, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_miou = 0
    step = start_epoch
    source_label = 0
    target_label = 1
    for epoch in range(start_epoch,args.num_epochs):
        #print(epoch)
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        discr_lr = poly_lr_scheduler(optimizer_D1,args.lr_discr,iter=epoch,max_iter=args.num_epochs)

        model.train()
        model_D1.train()


        tq = tqdm(total=len(dataloader_target) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_discr %f' % (epoch, lr,discr_lr))
        
        loss_record = []
        loss_source_record = []
        loss_target_record = []
        
        for i, ((src_x, src_y), (trg_x, _)) in enumerate(zip(dataloader_source, dataloader_target)):
            trg_x = trg_x.cuda()
            src_x = src_x.cuda()
            src_y = src_y.long().cuda()
            #print(i)
            optimizer.zero_grad()
            optimizer_D1.zero_grad()

            #Train G

            #Train with source
            for param in model_D1.parameters():
                param.requires_grad = False

        
            with amp.autocast():
                output_s, out16_s, out32_s = model(src_x)
                loss1 = loss_func(output_s, src_y.squeeze(1))
                loss2 = loss_func(out16_s, src_y.squeeze(1))
                loss3 = loss_func(out32_s, src_y.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            #print(loss)
            #Train with target

            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()

            with amp.autocast():
                output_t, out16_t, out32_t = model(trg_x)
                
                D_out1 = model_D1(F.softmax(output_t,dim=1))

                loss_adv_target1 = bce_loss(D_out1,torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

                loss_f = args.lambda_d1*loss_adv_target1

                loss = loss + loss_f

            scaler.scale(loss_f).backward()
            #scaler.step(optimizer)
            #scaler.update()
            #print(loss)
            #Train D
            loss_record.append(loss.item())
            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()


            for param in model_D1.parameters():
                param.requires_grad = True

            output_t = output_t.detach()
            output_s = output_s.detach()

            with amp.autocast():
                D_out1_s = model_D1(F.softmax(output_s,dim=1))
                loss_d1_s = bce_loss(D_out1_s,torch.FloatTensor(D_out1_s.data.size()).fill_(source_label).cuda())
            
            scaler.scale(loss_d1_s).backward()
            
            
            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()
            with amp.autocast():
                D_out1_t = model_D1(F.softmax(output_t,dim=1))
                loss_d1_t = bce_loss(D_out1_t,torch.FloatTensor(D_out1_t.data.size()).fill_(target_label).cuda())    
            
            scaler.scale(loss_d1_t).backward()
            
            
            scaler.step(optimizer_D1)
            scaler.step(optimizer)
            scaler.update()


            tq.update(args.batch_size)

            

            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            #writer.add_scalar('loss_step', loss, step)
            
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
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))
            torch.save({'state_dict':model_D1.state_dict(),'optimizer_state_dict': optimizer_D1.state_dict()}, os.path.join(args.save_model_path, 'latest_discr_'+str(epoch)+'.pth'))
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
    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou val', miou, epoch)


def train_improvements(args, model, model_D1, optimizer,optimizer_D1, dataloader_source, dataloader_target, dataloader_val, start_epoch, L, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_miou = 0
    step = start_epoch
    source_label = 0
    target_label = 1
    for epoch in range(start_epoch,args.num_epochs):
        #print(epoch)
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        discr_lr = poly_lr_scheduler(optimizer_D1,args.lr_discr,iter=epoch,max_iter=args.num_epochs)

        model.train()
        #model_D1.train()


        tq = tqdm(total=len(dataloader_target) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_discr %f' % (epoch, lr,discr_lr))
        
        loss_record = []
        loss_source_record = []
        loss_target_record = []
        
        for i, ((src_img, src_lbl), (trg_img, _)) in enumerate(zip(dataloader_source, dataloader_target)):
           
            
            scr_img_copy=src_img.clone()
            mean_img = torch.zeros(1,1)
            if mean_img.shape[-1] < 2:
                B, C, H, W = src_img.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
                
       
            # 1. source to target, target to target
            
            src_in_trg = FDA_source_to_target( src_img, trg_img, L)             # scr,src_lbl
            trg_in_trg = trg_img.cuda()                                              # trg, trg_lbl
           
            
            # 2. subtract mean
            #src_x = src_in_trg.clone() - mean_img                             # src, src_lbl
            #writer.add_image('eval%d/iter%d/predicted_eval_labels' % (epoch, i), np.array(src_in_trg), step, dataformats='NCHW')
            #trg_x = trg_in_trg.clone() - mean_img                             # trg, trg_lbl      
            #print(i)
            optimizer.zero_grad()
            #optimizer_D1.zero_grad()
            src_img= src_in_trg.cuda()
            src_lbl= src_lbl.long().cuda()
            #trg_x = trg_x.cuda()
            #src_x = src_x.cuda()
            #src_y = src_y.long().cuda()
            #Train G

            #Train with source
            #for param in model_D1.parameters():
            #    param.requires_grad = False

        
            with amp.autocast():
                output_s, out16_s, out32_s = model(src_img.half())
                loss1 = loss_func(output_s, src_lbl.squeeze(1))
                loss2 = loss_func(out16_s, src_lbl.squeeze(1))
                loss3 = loss_func(out32_s, src_lbl.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            '''
            with amp.autocast():
                output_s, out16_s, out32_s = model(trg_img)
                loss1 = loss_func(output_s, src_lbl.squeeze(1))
                loss2 = loss_func(out16_s, src_lbl.squeeze(1))
                loss3 = loss_func(out32_s, src_lbl.squeeze(1))
                lossT = loss1 + loss2 + loss3
            scaler.scale(lossT).backward()
            #scaler.step(optimizer)
            #scaler.update()
            #print(loss)
            #Train D
            loss_record.append(loss.item())
            '''
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
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))

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



            #scaler.step(optimizer)
            #scaler.update()

            #print(loss)
            #Train with target

            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()

'''           with amp.autocast():
                output_t, out16_t, out32_t = model(trg_x)
                
                D_out1 = model_D1(F.softmax(output_t,dim=1))

                loss_adv_target1 = bce_loss(D_out1,torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

                loss_f = args.lambda_d1*loss_adv_target1

                loss = loss + loss_f

            scaler.scale(loss_f).backward()
            #scaler.step(optimizer)
            #scaler.update()
            #print(loss)
            #Train D
            loss_record.append(loss.item())
            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()


            for param in model_D1.parameters():
                param.requires_grad = True

            output_t = output_t.detach()
            output_s = output_s.detach()

            with amp.autocast():
                D_out1_s = model_D1(F.softmax(output_s,dim=1))
                loss_d1_s = bce_loss(D_out1_s,torch.FloatTensor(D_out1_s.data.size()).fill_(source_label).cuda())
            
            scaler.scale(loss_d1_s).backward()
            
            
            #optimizer.zero_grad()
            #optimizer_D1.zero_grad()
            with amp.autocast():
                D_out1_t = model_D1(F.softmax(output_t,dim=1))
                loss_d1_t = bce_loss(D_out1_t,torch.FloatTensor(D_out1_t.data.size()).fill_(target_label).cuda())    
            
            scaler.scale(loss_d1_t).backward()
            
            
            scaler.step(optimizer_D1)
            scaler.step(optimizer)
            scaler.update()


            tq.update(args.batch_size)

            

            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            #writer.add_scalar('loss_step', loss, step)
            
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
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_model_path, 'latest_'+str(epoch)+'.pth'))
            torch.save({'state_dict':model_D1.state_dict(),'optimizer_state_dict': optimizer_D1.state_dict()}, os.path.join(args.save_model_path, 'latest_discr_'+str(epoch)+'.pth'))
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
    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou val', miou, epoch)'''
