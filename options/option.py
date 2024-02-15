import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--switch2entropy", 
                       type=int, 
                       default=16, 
                       help="switch to entropy after this many steps"
    )
    parse.add_argument("--entW",
                        type=float, 
                        default=0.005, 
                        help="weight for entropy")
    parse.add_argument("--ita", 
                       type=float, 
                       default=2.0, 
                       help="ita for robust entropy")
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
                          help='CityScapes, GTA5, CROSS_DOMAIN, DA or FDA . Define on which dataset the model should be trained and evaluated.')
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
    parse.add_argument('--lr_discr',
                       type=float,
                       default=0.0003,
                       help='Select if you want to resume from best or latest checkpoint')
    parse.add_argument('--lambda_d1',
                       type=float,
                       default=0.002,
                       help='Select if you want to resume from best or latest checkpoint')
    
    parse.add_argument('--l',
                       type=float,
                       default=0.001,
                       help='Set L of fourier transform')
    return parse.parse_args()