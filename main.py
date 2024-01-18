from model.model_stages import BiSeNet
from model.discriminator import FCDiscriminator
from dataset.cityscapes import CityScapes
from dataset.GTA5 import GTA5
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

from training.train import *
from validation.validate import *



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
                       default=8, #2
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
    parse.add_argument('--best',
                       type=str2bool,
                       default=False,
                       help='Select if you want to resume from best or latest checkpoint')
    parse.add_argument('--local',
                        type=str2bool,
                        default=True,
                        help='Select if you want to train on local or on colab')
    return parse.parse_args()

CITYSCAPES_CROPSIZE = (512,1024)
GTA_CROPSIZE = (720,1280)


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    

    
    print(args.dataset)
    print("Dim batch_size")
    print(args.batch_size)
    if args.local:
        initial_path = "."   
    else:
        initial_path = "/content"
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        
        transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()])
        
        train_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations)#eval_transformations)



    elif args.dataset == 'GTA5':
        print('training on GTA5')
        
        if args.augmentation:
            print("Performing data augmentation")
            transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
            train_dataset_big = GTA5(root = Path("/content"), transforms=transformations)
        else: 
            transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
            train_dataset_big = GTA5(root = Path("/content"), transforms=transformations)
        
        indexes = range(0, len(train_dataset_big))
        
        splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
        train_indexes = splitting[0]
        val_indexes = splitting[1]
        train_dataset = Subset(train_dataset_big, train_indexes)

        if args.augmentation:
            transformations = ExtCompose([ExtToTensor()])
            val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations)
        else:
            val_dataset = Subset(train_dataset_big, val_indexes)

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on CROSS_DOMAIN, training on GTA5 and validating on CityScapes')
        
        transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
        train_dataset = GTA5(root = Path("/content"), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations) 
    


    elif args.dataset == 'DA':
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
        
         #resize diversa per test
        transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()]) 
        target_dataset = CityScapes(root = "/content/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
        source_dataset = GTA5(root = Path("/content"), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= "/content/Cityscapes/Cityspaces", split='val',transforms=transformations)

        dataloader_source = DataLoader(source_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)


        dataloader_target = DataLoader(target_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)

        model_D1.train()
        model_D1.cuda()

        

        optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))



    else:
        print("Error, select a valid dataset")
        return None
    
    if args.dataset != 'DA':
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
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