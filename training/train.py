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

