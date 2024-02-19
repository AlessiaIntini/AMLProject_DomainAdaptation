import torch.nn as nn
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as FN
import numpy as np



def extract_ampl_phase(fft_re, fft_im):
    """
    Args:
        - fft_re: real part of fft
        - fft_im: imaginary part of fft
    return:
        - fft_amp: amplitude of fft
        - fft_pha: phase of fft
    """
    fft_amp = fft_re**2 + fft_im**2
    fft_amp = torch.sqrt(fft_amp)
    fft_complex = torch.complex(fft_re, fft_im)
    fft_pha = torch.angle(fft_complex)
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L ):
    """
    Args:
        - amp_src: amplitude of source fft
        - amp_trg: amplitude of target fft
        - L: the hyperparameter for low frequency mutation
    return:
        - amp_src: the mutated amplitude of source fft
    """
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b

    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L):
    """
    This function performs Fourier Domain Adaptation from source to target
    Args:
        - src_img: source image
        - trg_img: target image
        - L: the hyperparameter for low frequency mutation
    return:
        - src_in_trg: the source image adapted to the target style
    """

    # get fft of both source and target 
    fft_src_result = torch.fft.fftn( src_img,dim=(-4,-3,-2,-1))
    fft_re=fft_src_result.real
    fft_im=fft_src_result.imag
    fft_trg_result = torch.fft.fftn( trg_img,dim=(-4,-3,-2,-1))
    fft_trg_re=fft_trg_result.real
    fft_trg_im=fft_trg_result.imag

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_re.clone(),fft_im.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg_re.clone(),fft_trg_im.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src, amp_trg, L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src_result.size(), dtype=torch.complex64)
    fft_src_.real = torch.cos(pha_src.real.clone()) * amp_src_.clone()
    fft_src_.imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_=torch.complex(fft_src_.real,fft_src_.imag)
    src_in_trg = torch.fft.ifftn( fft_src_,dim=(-4,-3,-2,-1) ).real
    
    #normalize and scale the image to obtain the final result
    src_in_trg -= torch.min(src_in_trg)
    src_in_trg /= torch.max(src_in_trg)
    src_in_trg = (src_in_trg * 255).byte()

    return src_in_trg

class EntLoss(nn.Module):
    def __init__(self):
        super(EntLoss, self).__init__()

    def forward(self, x, ita):
        """
        Computes entropy loss for fourier domain adaptation

        Args:
        - x: input tensor
        - ita: hyperparameter to control the amount of entropy minimization
        return:
        - ent_loss_value: entropy loss value
        """

        P = FN.softmax(x, dim=1)        
        logP = FN.log_softmax(x, dim=1) 
        PlogP = P * logP               
        ent = -1.0 * PlogP.sum(dim=1)  
        ent = ent / 2.9444    
             
        # compute robust entropy
        ent = ent ** 2.0 + 1e-8
        ent = ent ** ita
        ent_loss_value = ent.mean()

        return ent_loss_value