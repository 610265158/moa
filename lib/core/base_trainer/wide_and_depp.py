
import torch
import torch.nn as nn

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


BN_MOMENTUM=0.02
BN_EPS=1e-5
ACT_FUNCTION=MemoryEfficientSwish


class Attention(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(Attention, self).__init__()

        self.att=nn.Sequential(nn.Linear(input_dim, output_dim//4,bias=False),
                               nn.BatchNorm1d(output_dim//4,momentum=BN_MOMENTUM,eps=BN_EPS),
                               ACT_FUNCTION(),
                               nn.Linear(output_dim//4, output_dim, bias=False),
                               nn.BatchNorm1d(output_dim, momentum=BN_MOMENTUM,eps=BN_EPS),
                               nn.Sigmoid())

    def forward(self, x):
        xx = self.att(x)

        return x*xx




class ResBlock(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(ResBlock, self).__init__()

        self.res=nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                               nn.BatchNorm1d(output_dim, momentum=BN_MOMENTUM,eps=BN_EPS),

                               )
        self.act=ACT_FUNCTION()

        self.att=Attention(input_dim=output_dim,output_dim=output_dim)


    def forward(self, x):
        xx = self.res(x)
        y=self.act(x + xx)

        y=self.att(y)
        return y



class Deep(nn.Module):

    def __init__(self, num_features=875, num_targets=206,num_extra_targets=402, hidden_size=512):
        super(Deep, self).__init__()

        self.bn_init = nn.BatchNorm1d(num_features, momentum=0.01, eps=BN_EPS)

        self.dense1 =nn.Sequential(nn.Linear(num_features, hidden_size,bias=False),
                                   nn.BatchNorm1d(hidden_size,momentum=BN_MOMENTUM,eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   )

        self.dense2 =nn.Sequential(ResBlock(hidden_size,hidden_size),
                                   nn.Dropout(0.5),
                                   ResBlock(hidden_size, hidden_size),
                                   nn.Dropout(0.5),
                                   ResBlock(hidden_size, hidden_size),
                                   nn.Dropout(0.5),
                                   )

        self.dense3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.bn_init(x)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.dense3(x)

        return x

class Wide(nn.Module):

    def __init__(self, num_features=875, num_targets=206,num_extra_targets=402, hidden_size=512):
        super(Wide, self).__init__()

        self.bn_init = nn.BatchNorm1d(num_features, momentum=0.01, eps=BN_EPS)

        self.dense1 =nn.Sequential(nn.Linear(num_features, hidden_size,bias=False),
                                   nn.BatchNorm1d(hidden_size,momentum=BN_MOMENTUM,eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_size, hidden_size, bias=False),
                                   nn.BatchNorm1d(hidden_size, momentum=BN_MOMENTUM, eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   )

        self.dense3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.bn_init(x)

        x = self.dense1(x)
        x = self.dense3(x)

        return x

class WideAndDeep(nn.Module):

    def __init__(self, num_features=875, num_targets=206,num_extra_targets=402, hidden_size=512):
        super(WideAndDeep, self).__init__()


        self.wide=Wide(num_features=num_features,hidden_size=hidden_size*2)
        self.deep=Deep(num_features=num_features,hidden_size=hidden_size)


        self.att=Attention(hidden_size*3,hidden_size*3)

        self.dense4 = nn.Linear(hidden_size * 3, num_targets)

        self.dense5 = nn.Linear(hidden_size * 3, num_extra_targets)
    def forward(self, x):
        xx = self.wide(x)
        yy = self.deep(x)

        feature=torch.cat([xx,yy],dim=1)

        xx = self.dense4(feature)
        yy = self.dense5(feature)
        return xx,yy

