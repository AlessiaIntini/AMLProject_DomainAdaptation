#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from model.discriminator import FCDiscriminator
from cityscapes import CityScapes
from GTA5 import GTA5
import torchvision.transforms as transforms
from torchvision.transforms import v2
from utils import ExtCompose, ExtResize, ExtToTensor, ExtTransforms, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop, ExtColorJitter
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


logger = logging.getLogger()


def val(args, model, dataloader, writer = None , epoch = None, step = None):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        random_sample = [random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1),
                        random.randint(0, len(dataloader) - 1)]
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            
            if i in random_sample and writer is not None:
                if args.dataset == 'CITYSCAPES':
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(predict, label)
                elif args.dataset == 'GTA5':
                    colorized_predictions , colorized_labels = GTA5.visualize_prediction(predict, label)
                elif args.dataset == 'CROSS_DOMAIN':
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(predict, label)    
                writer.add_image('eval%d/iter%d/predicted_eval_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/correct_eval_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/eval_original _data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')

                #colorized_predictions.save("/content/results/img"+str(i)+".png")
                #colorized_labels.save("/content/results/lbl"+str(i)+".png")

                #import matplotlib.pyplot as plt
                #plt.imshow("/content/results/img"+str(i)+".png")
                #plt.imshow("/content/results/lbl"+str(i)+".png")


            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

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
    val(args, model, dataloader_val, writer, epoch, step)


def train_and_adapt(args, model, model_D1, optimizer,optimizer_D1, dataloader_source, dataloader_target, dataloader_val, start_epoch, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) 
    bce_loss = torch.nn.MSELoss()
    max_miou = 0
    step = start_epoch
    source_label = 0
    target_label = 1
    for epoch in range(start_epoch,args.num_epochs):
        print(epoch)
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        discr_lr = poly_lr_scheduler(optimizer_D1,0.001,iter=epoch,max_iter=args.num_epochs)

        model.train()
        model_D1.train()


        tq = tqdm(total=len(dataloader_source) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_discr %f' % (epoch, lr,discr_lr))
        #print(len(tuple(zip(dataloader_source, dataloader_target))))
        loss_record = []
        #image_number = random.randint(0, len(dataloader_train) - 1)
        for i, ((src_x, src_y), (trg_x, _)) in enumerate(zip(dataloader_source, dataloader_target)):
            trg_x = trg_x.cuda()
            src_x = src_x.cuda()
            src_y = src_y.long().cuda()
            print(i)
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
            scaler.step(optimizer)
            scaler.update()

            print(loss)
            #Train with target
            with amp.autocast():
                output_t, out16_t, out32_t = model(trg_x)

                D_out1 = model_D1(F.softmax(out32_t))

                loss_adv_target1 = bce_loss(D_out1,Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

                loss = 0.1*loss_adv_target1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(loss)
            #Train D
            for param in model_D1.parameters():
                param.requires_grad = True

            D_out1_s = model_D1(F.softmax(out32_s))
            loss_d1_s = bce_loss(D_out1_s,Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            scaler.scale(loss_d1_s).backward()
            
            
            D_out1_t = model_D1(F.softmax(out32_t))
            loss_d1_t = bce_loss(D_out1_t,Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())    
            scaler.scale(loss_d1_t).backward()
            
            
            scaler.step(optimizer_D1)
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
    val(args, model, dataloader_val, writer, epoch, step)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
    )

    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='STDCNet813',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='checkpoints/STDCNet813M_73.91.tar',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,#300
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=1,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=5,
                       help='How often to perform validation (epochs)')
    
    parse.add_argument('--batch_size',
                       type=int,
                       default=4, #2
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01, #0.01
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=2, #4
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,#19
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default='checkpoints',
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    parse.add_argument('--resume',
                       type=str2bool,
                       default=False,
                       help='Define if the model should be trained from scratch or from a trained model')
    parse.add_argument('--dataset',
                          type=str,
                          default='CityScapes',
                          help='CityScapes, GTA5 or CROSS_DOMAIN. Define on which dataset the model should be trained and evaluated.')
    parse.add_argument('--resume_model_path',
                       type=str,
                       default='',
                       help='Define the path to the model that should be loaded for training. If void, the last model will be loaded.')
    parse.add_argument('--comment',
                       type=str,
                       default='',
                       help='Optional comment to add to the model name and to the log.')
    parse.add_argument('--augmentation',
                       type=str2bool,
                       default=False,
                       help='Select if you want to perform some data augmentation')
    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    
    
    print(args.dataset)
    print("Dim batch_size")
    print(args.batch_size)
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        cropsize = (512,1024) #resize diversa per test
        transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
        
        train_dataset = CityScapes(root = "/content/Cityscapes/Cityspaces", split = 'train',transforms=transformations)
        val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations)#eval_transformations)

    elif args.dataset == 'GTA5':
        print('training on GTA5')
        cropsize = (720,1280)
        #eval_transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
        if args.augmentation:
            print("Performing data augmentation")
            transformations = ExtCompose([ExtRandomCrop(cropsize), ExtRandomHorizontalFlip(), ExtToTensor()])
            train_dataset_big = GTA5(root = Path("/content"), transforms=transformations)
        else: 
            transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
            train_dataset_big = GTA5(root = Path("/content"), transforms=transformations)
        
        indexes = range(0, len(train_dataset_big))
        
        splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
        train_indexes = splitting[0]
        val_indexes = splitting[1]
        train_dataset = Subset(train_dataset_big, train_indexes)
        val_dataset = Subset(train_dataset_big, val_indexes)
    elif args.dataset == 'DA':
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
        
        cropsize = (512,1024) #resize diversa per test
        transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
        
        target_dataset = CityScapes(root = "/content/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        cropsize = (720,1280)
        transformations = ExtCompose([ExtRandomCrop(cropsize), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
        source_dataset = GTA5(root = Path("/content"), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations)#eval_transformations)

        dataloader_source = DataLoader(source_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)


        dataloader_target = DataLoader(target_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)

        model_D1.train()
        model_D1.cuda()

        

        optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        #optimizer_D1.zero_grad()


        bce_loss = torch.nn.BCEWithLogitsLoss()


    else:
        print('training on CROSS_DOMAIN, training on GTA5 and validating on CityScapes')
        cropsize = (720,1280)
        transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
        #transformations = ExtCompose([ExtToTensor()])
        train_dataset = GTA5(root = Path("/content"), transforms=transformations)
        cropsize = (512,1024)
        #transformations = ExtCompose([ExtResize(cropsize), ExtToTensor()])
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations) 
    
    #dataloader_train = DataLoader(train_dataset,
    #                batch_size=args.batch_size,
    #                shuffle=False,
    #                num_workers=args.num_workers,
    #                pin_memory=False,
    #                drop_last=True)
    dataloader_val = DataLoader(val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)
    
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    
    start_epoch = 0
    
    if args.resume:
        for check in os.listdir('./checkpoints'):
            if 'latest_' in check:

                start_epoch_tmp = int(check.split('_')[1].replace('.pth',''))

                if start_epoch_tmp >= start_epoch:
                    start_epoch = start_epoch_tmp+1
                    pretrain_path = "checkpoints/"+check

        #if args.resume and "latest_" in os.listdir("./checkpoints"):
        #    model

        if start_epoch > 0:
            checkpoint = torch.load(pretrain_path)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded latest checkpoint")


    
    match args.mode:
        case 'train':
            ## train loop
            train(args, model, optimizer, dataloader_train, dataloader_val,start_epoch, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case 'test':
            writer = SummaryWriter(comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
            val(args, model, dataloader_val, writer=writer,epoch=0,step=0)
        case 'adapt':
            train_and_adapt(args, model,model_D1, optimizer,optimizer_D1, dataloader_source,dataloader_target, dataloader_val,start_epoch, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case _:
            print('not supported mode \n')
            return None


if __name__ == "__main__":
    main()