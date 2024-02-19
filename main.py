from model.model_stages import BiSeNet
from model.discriminator import FCDiscriminator
from dataset.cityscapes import CityScapes
from dataset.GTA5 import GTA5
import torchvision.transforms as transforms
from utils.utils import *
from utils.augUtils import *
import torch
from torch.utils.data import DataLoader, Subset
from options.option import parse_args
import numpy as np
from tensorboardX import SummaryWriter
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from training.train import *
from training.trainADA import *
from training.trainFDA import *
from validation.validate import *

CITYSCAPES_CROPSIZE = (512,1024)
GTA_CROPSIZE = (720,1280)
    
def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    
    print("Dataset " + args.dataset)
    print("Dim batch_size " + str(args.batch_size))
    print("Optimizer is " + args.optimizer)

    # If the script is running on colab use local = false
    if args.local:
        initial_path = "."   
    else:
        initial_path = "/content"

    # Training on Cityscapes as train and validation dataset
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        
        # For the train just resize the images to obtain a faster training
        transformations = ExtCompose([ExtResize(CITYSCAPES_CROPSIZE), ExtToTensor()])    
        train_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        # For validation use the full size images from the val dataset
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations)#eval_transformations)


    # Training on GTA as train dataset and GTA5 as val dataset if no data augmentation is performed
    # otherwise on Cityscapes as validation
    elif args.dataset == 'GTA5':
        print('training on GTA5')
        
        if args.augmentation:
            print("Performing data augmentation")
            #GTA5 used as source and trasformations of data augmentation are used
            #transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(),ExtColorJitter(p=0.5,brightness=0.2,contrast= 0.3,saturation= 0.3,hue= 0.4), ExtGaussianBlur(), ExtToTensor()])
            #best data augmentation configurations
            transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(),ExtColorJitter(p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2), ExtToTensor()])
            train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
        else: 
            # If no augmentation just resize the images
            transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
            train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
        
        #Create an array containing all the index of the dataset
        indexes = range(0, len(train_dataset_big))
        
        # Split the indexes with shuffle = true, we are using 75% of the dataset for train
        # and 25% for validation
        splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
        train_indexes = splitting[0]
        train_dataset = Subset(train_dataset_big, train_indexes)

        # If augmentation take Cityscapes as validation dataset
        if args.augmentation:
            transformations = ExtCompose([ExtToTensor()])
            val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations)
        else:
            transformations = ExtCompose([ExtToTensor()])
            #train_dataset_big = GTA5(root = Path(initial_path), transforms=transformations)
            #indexes = range(0, len(train_dataset_big))
            #splitting = train_test_split(indexes, train_size = 0.75, random_state = 42, shuffle = True)
            val_indexes = splitting[1]
            val_dataset = Subset(train_dataset_big, val_indexes)


    # Training on GTA5 and validating on Cityscapes
    elif args.dataset == 'CROSS_DOMAIN':
        #For this section is possible also to use the training result on GTA5 and validate on Cityscapes
        print('training on CROSS_DOMAIN, training on GTA5 and validating on CityScapes')
        #Training on all GTA5 dataset
        transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
        train_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        transformations = ExtCompose([ExtToTensor()])
        val_dataset = CityScapes(root= initial_path + "/Cityscapes/Cityspaces", split='val',transforms=transformations) 


    # Training on GTA5 using adversarial domain adaptation, validating on Cityscapes
    elif args.dataset == 'DA':
        
        # Create the discriminator model
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
        
        # Cityscapes used as target
        transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()]) 
        target_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)
        
        # GTA as source
        transformations = ExtCompose([ExtRandomCrop(GTA_CROPSIZE), ExtRandomHorizontalFlip(),ExtColorJitter(p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2), ExtToTensor()])
        source_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        #Cityscapes used as validation
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

        # Choose the optimizer for the discriminator
        if args.optimizer == 'rmsprop':
            optimizer_D1 = torch.optim.RMSprop(model_D1.parameters(), lr=args.lr_discr)
        elif args.optimizer == 'sgd':
            optimizer_D1 = torch.optim.SGD(model_D1.parameters(), lr=args.lr_discr, momentum=0.9, weight_decay=1e-4)
        elif args.optimizer == 'adam':
            optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.lr_discr)
        else:  
            print('not supported optimizer \n')
            return None


    # Training on GTA using Fourier domain adaptation
    elif args.dataset == 'FDA':
        print('training on FDA')
        
        # Cityscapes as target (using for both the dataloader GTA_CROPSIZE to align the images)
        transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()]) 
        target_dataset = CityScapes(root = initial_path + "/Cityscapes/Cityspaces", split = 'train',transforms=transformations)

        # GTA5 as source
        transformations = ExtCompose([ExtResize(GTA_CROPSIZE), ExtToTensor()])
        source_dataset = GTA5(root = Path(initial_path), transforms=transformations)
        
        # Cityscapes as validation
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
    

    # Creating the model, with backbone STDC passed as argument
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else: 
        print('not supported optimizer \n')
        return None
    
    start_epoch = 0
    
    # If resume = true, take the last checkpoint saved in the validation phase, if dataset = DA
    # also the discriminator must be resumed 
    if args.resume and args.dataset != 'DA':
        for check in os.listdir('./checkpoints'):
            if 'latest_' in check:

                start_epoch_tmp = int(check.split('_')[1].replace('.pth',''))

                if start_epoch_tmp >= start_epoch:
                    start_epoch = start_epoch_tmp+1
                    pretrain_path = "checkpoints/"+check

        if start_epoch > 0:
            print(pretrain_path)
            checkpoint = torch.load(pretrain_path)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded latest checkpoint")

    # If dataset != DA just resume the model and the optimizer
    elif args.resume:
        for check in os.listdir('./checkpoints'):
            if 'latest_discr_' in check:

                start_epoch_tmp = int(check.split('_')[2].replace('.pth',''))

                if start_epoch_tmp >= start_epoch:
                    start_epoch = start_epoch_tmp+1
                    pretrain_discr_path = "checkpoints/"+check
                    pretrain_path = "checkpoints/latest_"+str(start_epoch_tmp)+".pth"
            
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
            # Adversarial domain adaptation
            train_ADA(args, model,model_D1, optimizer,optimizer_D1, dataloader_source,dataloader_target, dataloader_val,start_epoch, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case 'improvements':
            # Fourier Domain Adaptation
            print("L value is " + str(args.l))
            train_FDA(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val,start_epoch, L=args.l, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))    
        case _:
            print('not supported mode \n')
            return None


if __name__ == "__main__":
    main()