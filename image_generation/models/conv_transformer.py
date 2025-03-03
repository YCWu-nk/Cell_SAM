import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

class TransformerGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(TransformerGenerator, self).__init__()
        self.model = nn.Transformer(d_model=ngf, nhead=2, num_encoder_layers=2, num_decoder_layers=2)
        self.fc = nn.Linear(ngf, output_nc)

    def forward(self, src, tgt, z):
        output = self.model(src, tgt)
        output = self.fc(output)
        return output

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_nc, ngf):
        super(TransformerFeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_nc, ngf)

    def forward(self, input):
        output = self.fc(input)
        return output

def de_G(input_nc, output_nc, ngf, netG, normG, dropout, init_type, init_gain, no_antialias, no_antialias_up, gpu_ids, opt):
    net = None
    if netG == 'transformer':
        net = TransformerGenerator(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model [%s] is not implemented' % netG)
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = nn.DataParallel(net, gpu_ids)

    net.apply(weights_init(init_type, init_gain))

    return net

def de_F(input_nc, netF, normG, dropout, init_type, init_gain, no_antialias, gpu_ids, opt):
    net = None
    if netF == 'transformer':
        net = TransformerFeatureExtractor(input_nc, opt.netF_nc)
    else:
        raise NotImplementedError('Feature extractor model [%s] is not implemented' % netF)
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = nn.DataParallel(net, gpu_ids)

    net.apply(weights_init(init_type, init_gain))

    return net

def weights_init(init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    return init_func