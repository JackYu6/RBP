import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,\
    wide_resnet50_2, wide_resnet101_2


class RBP(nn.Module):
    def __init__(self, num_classes, backbone=resnet50(), dr=True, dr_n=2):
        super(RBP, self).__init__()
        self.dr = dr
        if not dr:
            dr_n = 1
        self.features = nn.Sequential(backbone.conv1,
                                      backbone.bn1,
                                      backbone.relu,
                                      backbone.maxpool,
                                      backbone.layer1,
                                      backbone.layer2,
                                      backbone.layer3,
                                      backbone.layer4)
        # dimension reduction 降维
        self.dr_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone.fc.in_features,
                out_channels=backbone.fc.in_features // dr_n,  # 注意要写成整除，保证数据类型为int
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(backbone.fc.in_features // dr_n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.classifiers = nn.Sequential(nn.Linear((backbone.fc.in_features // dr_n) ** 2, num_classes))
        # resnet18、34的in_features为512, resnet50及以上为2048，实际上就是经过最后一层卷积层后通道数的大小

    def forward(self, x):
        x = self.features(x)
        if self.dr:
            x = self.dr_conv(x)
        batch_size = x.size(0)
        channel_num = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_num, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5))
        x = self.classifiers(x)
        return x


if __name__ == '__main__':
    model = RBP(num_classes=10)
    print(model)
