from dataset.cityscapes import CityScapes
from dataset.GTA5 import GTA5
import torch
import numpy as np
from tensorboardX import SummaryWriter
from utils.utils import *
import random

#Validation function used to claculate the metrics on the test/validation dataset
def val(args, model, dataloader, writer = None , epoch = None, step = None):
    print('start val!')
    
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        #take some random index to visualize a sample of the predictions
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
        # Iterate over the test dataloader
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            # Save the image if it is included in the sample
            if i in random_sample and writer is not None:
                if args.dataset == 'CITYSCAPES' or args.dataset == 'DA' or args.dataset == 'CROSS_DOMAIN' or args.augmentation or args.dataset == 'FDA' :
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(predict, label)
                elif args.dataset == 'GTA5':
                    colorized_predictions , colorized_labels = GTA5.visualize_prediction(predict, label)
                
                # Save the images to tensorboard
                writer.add_image('eval%d/iter%d/predicted_eval_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/correct_eval_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/eval_original _data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')

                
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            
            precision_record.append(precision)
            
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')
        writer.add_scalar('epoch/precision_val', precision, epoch)
        writer.add_scalar('epoch/miou val', miou, epoch)

        return precision, miou
