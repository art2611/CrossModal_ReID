import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import resnet50


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_t = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.visible = model_t


    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x

class shared_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(shared_resnet, self).__init__()

        model_base = resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_base.fc = Identity()
        self.base = model_base

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class Network_fuse(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(Network_fuse, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.shared_resnet = shared_resnet(arch=arch)

        pool_dim = 2048

        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)    #torch.Size([32, 2048, 18, 9])
            x2 = self.thermal_module(x2)    #torch.Size([32, 2048, 18, 9])
            x = torch.cat((x1, x2), 0)      #torch.Size([64, 2048, 18, 9])
            # print(x1.shape)
            # print(x2.shape)
            # print(x.shape)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.shared_resnet(x)

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool) #torch.Size([64, 2048])

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

class Resnet_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(Resnet_module, self).__init__()

        model_base = resnet50(pretrained=True)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_base.fc = Identity()
        # avg pooling to global pooling
        self.res = model_base


    def forward(self, x):
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x = self.res.relu(x)
        x = self.res.maxpool(x)
        x = self.res.layer1(x)
        x = self.res.layer2(x)
        x = self.res.layer3(x)
        x = self.res.layer4(x)
        return x

class Network(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(Network, self).__init__()
        model_base = resnet50(pretrained=True)

        self.Resnet_module = Resnet_module(arch=arch)
        pool_dim = 2048

        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x):

        x = self.Resnet_module(x)    #torch.Size([64, 2048, 18, 9])

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1)) # Incomprehension, on recupere la même chose

        feat = self.bottleneck(x_pool) #torch.Size([64, 2048])

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

