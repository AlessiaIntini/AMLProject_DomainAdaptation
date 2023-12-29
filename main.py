import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm


from model.model_stages import BiSeNet
from cityscapes import CityScapes
from gta5 import GTA5
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np

from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm

logger = logging.getLogger()


def val(model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((num_classes, num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    

import math
def train( model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(optimizer))
    nans = []
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(num_epochs):
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

           #print("\n")
           #print(data.size())
           #print("\n")
           #print(label.size())
           #print(torch.squeeze(label,1).size())
           #print("\n")

            with amp.autocast():
                output, out16, out32 = model(data)
                #print("###")
                #print(label.squeeze(1).size())
                #print("###")
                #print(out16.size())
                #print("###")
                #print(out32.size())
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3
                #print(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            if math.isnan(loss.item()):
              nans.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % checkpoint_step == 0 and epoch != 0:
            #print("\n")
            #print(nans)
            #print("\n")
            import os
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            torch.save({'state_dict':model.module.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_model_path, 'latest_'+str(epoch)+'.pth'))

        if epoch % validation_step == 0 and epoch != 0:
            precision, miou = val( model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


if __name__ == '__main__':
    optimizer = "adam"
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 5
    num_classes = 19
    checkpoint_step = 1
    save_model_path = "./my_checkpoint"
    validation_step = 10
    mode = "train"
    num_workers = 2
    backbone = "STDCNet813"
    pretrain_path = "checkpoints/STDCNet813M_73.91.tar"
    use_conv_last = False
    use_gpu = True
    n_classes = num_classes

    train_dataset = CityScapes(split='train',root='./Cityscapes/Cityspaces', target_type='color')
    dataloader_train = DataLoader(train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True)

    val_dataset = CityScapes(split='val',root='./Cityscapes/Cityspaces', target_type='color')
    dataloader_val = DataLoader(val_dataset,
                   batch_size=1,
                   shuffle=False,
                   num_workers=num_workers,
                   drop_last=False)
    
    ## model
    model = BiSeNet(backbone=backbone, n_classes=n_classes, pretrain_model=pretrain_path, use_conv_last=use_conv_last)

    if torch.cuda.is_available() and use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    else:  # rmsprop
        print('not supported optimizer, using adam\n')
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)


    ## train loop
    train(model, optimizer, dataloader_train, dataloader_val)
    # final test


    val(model, dataloader_val)