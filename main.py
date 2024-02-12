from model.model_stages import BiSeNet
from model.discriminator import FCDiscriminator
from dataset.cityscapes import CityScapes
from dataset.GTA5 import GTA5
import torchvision.transforms as transforms
from torchvision.transforms import v2
from utils import *
import torch
from torch.utils.data import DataLoader, Subset
import logging
from options.option import parse_args
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

CITYSCAPES_CROPSIZE = (512,1024)
GTA_CROPSIZE = (720,1280)

#def collate_fn(batch):
    # Estrai solo i vettori target da ogni elemento nel batch
   # targets = [item[1] for item in batch]

    # Converti la lista di target in un tensore PyTorch
   # targets = torch.stack(targets, dim=0)

   # return targets
    
def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    
    print("Dataset " + args.dataset)
    print("Dim batch_size " + str(args.batch_size))
    print("Optimizer is " + args.optimizer)

    if args.local:
        initial_path = "."   
    else:
        initial_path = "/content"
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        
        transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()])
        #transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor(),ExtNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        #transformations = ExtCompose([ExtScale(0.5), ExtToTensor()])
        
        train_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)
        #transformations = ExtCompose([ExtToTensor(),ExtNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations)#eval_transformations)



    elif args.dataset == 'GTA5':
        print('training on GTA5')
        
        if args.augmentation:
            print("Performing data augmentation")
            #transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
            transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
            train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
        else: 
            transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
            train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
        
        indexes = range(0, len(train_dataset_big))
        
        splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
        train_indexes = splitting[0]
        train_dataset = Subset(train_dataset_big, train_indexes)

        if args.augmentation:
            transformations = ExtCompose([ExtToTensor()])
            val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations)
        else:
            transformations = ExtCompose([ExtToTensor()])
            train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
            indexes = range(0, len(train_dataset_big))
            splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
            val_indexes = splitting[1]
            val_dataset = Subset(train_dataset_big, val_indexes)

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on CROSS_DOMAIN, training on GTA5 and validating on CityScapes')
        
        #transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
        transformations = ExtCompose([ExtScale(), ExtToTensor()])
        train_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations) 
    


    elif args.dataset == 'DA':
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
        
         #resize diversa per test
        #transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()]) 
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtToTensor()]) 
        target_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
        source_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='train',transforms=transformations)

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

        if args.optimizer == 'rmsprop':
            optimizer_D1 = torch.optim.RMSprop(model_D1.parameters(), lr=args.lr_discr)
        elif args.optimizer == 'sgd':
            optimizer_D1 = torch.optim.SGD(model_D1.parameters(), lr=args.lr_discr, momentum=0.9, weight_decay=1e-4)
        elif args.optimizer == 'adam':
            optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.lr_discr)
        else:  # rmsprop
            print('not supported optimizer \n')
            return None

    elif args.dataset == 'FDA':
        print('training on FDA')
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
        
         #resize diversa per test
        #transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()]) 
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtToTensor()]) 
        target_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        
        #transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(), ExtColorJitter(0.5,0.5,0.5,0.5), ExtToTensor()])
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtToTensor(),ExtRandomHorizontalFlip()])
        source_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='train',transforms=transformations)
    
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

        if args.optimizer == 'rmsprop':
            optimizer_D1 = torch.optim.RMSprop(model_D1.parameters(), lr=args.lr_discr)
        elif args.optimizer == 'sgd':
            optimizer_D1 = torch.optim.SGD(model_D1.parameters(), lr=args.lr_discr, momentum=0.9, weight_decay=1e-4)
        elif args.optimizer == 'adam':
            optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.lr_discr)
        else:  # rmsprop
            print('not supported optimizer \n')
            return None

    else:
        print("Error, select a valid dataset")
        return None
    
    if args.dataset != 'DA' and args.dataset != 'FDA':
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
    
    if args.resume and args.dataset != 'DA' and args.dataset != 'FDA':
        for check in os.listdir('./checkpoints'):
            if 'latest_' in check:

                start_epoch_tmp = int(check.split('_')[1].replace('.pth',''))

                if start_epoch_tmp >= start_epoch:
                    start_epoch = start_epoch_tmp+1
                    pretrain_path = "checkpoints/"+check

        #if args.resume and "latest_" in os.listdir("./checkpoints"):
        #    model

        if start_epoch > 0:
            print(pretrain_path)
            checkpoint = torch.load(pretrain_path)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded latest checkpoint")

    elif args.resume:
        for check in os.listdir('./checkpoints'):
            if 'latest_discr_' in check:

                start_epoch_tmp = int(check.split('_')[2].replace('.pth',''))

                if start_epoch_tmp >= start_epoch:
                    start_epoch = start_epoch_tmp+1
                    pretrain_discr_path = "checkpoints/"+check
                    pretrain_path = "checkpoints/latest_"+str(start_epoch_tmp)+".pth"
            

        #if args.resume and "latest_" in os.listdir("./checkpoints"):
        #    model

        if start_epoch > 0:
            print(pretrain_path)
            checkpoint = torch.load(pretrain_path)
            
            checkpoint_discr = torch.load(pretrain_discr_path)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.dataset == 'DA':
                model_D1.load_state_dict(checkpoint_discr['state_dict'])
                optimizer_D1.load_state_dict(checkpoint_discr['optimizer_state_dict'])
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
        case 'improvements':
            print("L value is " + str(args.l))
            train_improvements(args, model,model_D1, optimizer,optimizer_D1, dataloader_source,dataloader_target, dataloader_val,start_epoch, L=args.l, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))    
        case _:
            print('not supported mode \n')
            return None


if __name__ == "__main__":
    main()