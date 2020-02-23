import torch
import sys
sys.path.append('..')

from chamfer_dist import ChamferFunction

from torch.autograd import gradcheck

kwargs = {'dtype': torch.float32, 'device': torch.device("cuda"), 'requires_grad': True}

ptcloud = torch.randn(2, 1024, 3, **kwargs)
gtcloud = torch.randn(2, 16384, 3, **kwargs)
print(gradcheck(ChamferFunction.apply, [ptcloud, gtcloud]))
