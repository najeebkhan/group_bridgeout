import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.autograd import Variable


class GroupBridgeoutFcLayer(Module):
    r"""TODO
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(
             self,
            in_features,
            out_features,
            reg_strength=0.5,
            target_fraction=1.0,
            bias=True,
            **factory_kwargs):
        super(GroupBridgeoutFcLayer, self).__init__()
        
        assert target_fraction==1.0, 'not implemented'

        self.nu = reg_strength

        self.p=0.5
        self.target_fraction = target_fraction
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_x):
        if not self.training or self.nu <= 0.0:
            return F.linear(input_x, self.weight, self.bias) 

        batch_size =  input_x.numel()//input_x.size()[-1]
        regularization_strength = self.nu/batch_size

        w = self.weight
        # l2 raised to the power 0.5 so that the effective penalty is L1 of L2 norms.
        wq = torch.norm(w, 2, dim=1).mul(regularization_strength).pow( 0.5 ).unsqueeze(1)

        noise = w.data.clone()
        noise.bernoulli_(1 - self.p).div_(1 - self.p).sub_(1)
        targeting_mask = 1.0         

        perturbation_equivalent = wq.mul(Variable(noise)).mul(targeting_mask)
        w = w.add( perturbation_equivalent )
        output = F.linear(input_x, w, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
        
if __name__=='__main__':
    from copy import deepcopy
    torch.manual_seed(33)
    m = GroupBridgeoutFcLayer(in_features=2, out_features=3, p=0.2)

    inp = torch.tensor(
    [
        [1,2.],
        [3,4],
        [4,5],
    ])
    output = m(inp)
    print(output)
    
