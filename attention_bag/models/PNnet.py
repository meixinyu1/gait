import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

#
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

input_dim = 256

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # this changed
        # self.new_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2,bias=False)
        self.smaller_new_conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # this changed
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.smaller_new_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.new_fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.feature_softmax = nn.Sigmoid()

        self.n_layer_fr = self._make_n_layer(input_dim, 1)
        self.n_layer_ba = self._make_n_layer(input_dim, 1)
        self.n_layer_sm = self._make_n_layer(input_dim, 1)
        self.n_layer_sb = self._make_n_layer(input_dim, 1)
        self.n_layer_mu = self._make_n_layer(input_dim, 1)
        self.n_layer_cp = self._make_n_layer(input_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_n_layer(self, input_dim, output_dim):
        return nn.Sequential(
            # # #type 1
            # nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(output_dim),
            # nn.ReLU(inplace=True),

            #type 2
            nn.Conv2d(input_dim, input_dim//2, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(input_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim//2, output_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, tag=None):
        weight = 0.5
        N = True
        x = self.smaller_new_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.smaller_new_maxpool(x)

        x = self.layer1(x)
        if N:
            layer_bags = [x]*6
            n,c,h,w = x.size()
            tmp = self.n_layer_fr(layer_bags[0]).view(n,-1)
            layer_bags[0] = self.feature_softmax(tmp).view(n,1,h,w)
            layer_bags[1] = self.feature_softmax(self.n_layer_ba(layer_bags[1]).view(n,-1)).view(n,1,h,w)
            layer_bags[2] = self.feature_softmax(self.n_layer_sm(layer_bags[2]).view(n,-1)).view(n,1,h,w)
            layer_bags[3] = self.feature_softmax(self.n_layer_sb(layer_bags[3]).view(n,-1)).view(n,1,h,w)
            layer_bags[4] = self.feature_softmax(self.n_layer_mu(layer_bags[4]).view(n,-1)).view(n,1,h,w)
            layer_bags[5] = self.feature_softmax(self.n_layer_cp(layer_bags[5]).view(n,-1)).view(n,1,h,w)

            layer_bags = torch.stack(layer_bags)
            layer_bags = layer_bags.view(layer_bags.size(0), layer_bags.size(1), layer_bags.size(3), layer_bags.size(4))
            layer_bags = layer_bags.transpose(0, 1)

            if tag is not None:
                # _, index = layer_bags.view(layer_bags.size(0), layer_bags.size(1), -1).sum(2).max(1)
                # tt = []
                # for i in range(n):
                #     tt.append(layer_bags[i,index[i],:])
                # tt = torch.stack(tt).unsqueeze(1)
                tt = (layer_bags*tag.view(tag.size(0), tag.size(1), 1, 1)).sum(dim = 1, keepdim=True)
                x = x - tt*x*weight
                n = (layer_bags * (1 - tag.view(tag.size(0),tag.size(1), 1, 1)))
                p = (layer_bags*tag.view(tag.size(0), tag.size(1), 1, 1))
                n_loss = (n,p)
            else:
                _, index = layer_bags.view(layer_bags.size(0), layer_bags.size(1), -1).sum(2).max(1)
                tt = []
                for i in range(n):
                    tt.append(layer_bags[i,index[i],:])
                tt = torch.stack(tt).unsqueeze(1)
                x = x - tt*x*weight
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        x = self.new_fc(features)
        x = self.drop_out(x)
        if tag is None:
            return x, features
        if N:
            return x, features, n_loss
        else:
            return x, features


def resnet18(pretrained=False, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)

    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # load 部分参数
        preDict = torch.load("./save_model/resnet18.pth")
        modelDict = model.state_dict()
        preDict = {k: v for k, v in preDict.items() if k in modelDict}
        modelDict.update(preDict)
        model.load_state_dict(modelDict)
        # model.load_state_dict(torch.load("./save_model/resnet18-5c106cde.pth"))
    return model


def resnet50(pretrained=False, num_classes=1000, user = 'meixinyu',**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)
    if pretrained:

        # path1 = "/home/meixinyu/gait/save_model/resnet50.pth"
        # path = '/home/meixinyu/gait/attention_N_P/log1/checkpoint_ep10.pth.tar'
        if user == 'meixinyu':
            path1 = "/home/meixinyu/gait/save_model/resnet50.pth"
        elif user == "zhangx":
            path1 = "/home/zhangsx/gait/save_model/resnet50.pth"
        elif user == "loki":
            path1 = "/home/jky/loki/work/gait/save_model/resnet50.pth"
            path1 = '/home/jky/loki/work/gait/save_model/savePoint/local_features/checkpoint_ep83.pth.tar'
        preDict = torch.load(path1)
        # print(preDict['rank1'])
        # print(preDict['state_dict'])
        # preDict = preDict['state_dict']
        modelDict = model.state_dict()
        preDict = {k: v for k, v in preDict.items() if k in modelDict}
        modelDict.update(preDict)
        model.load_state_dict(modelDict)
    return model
