import torch
import torch.nn as nn
from torchvision.models resnet50


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
        self.dr_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone.fc.in_features,
                out_channels=backbone.fc.in_features // dr_n,  
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(backbone.fc.in_features // dr_n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(nn.Linear((backbone.fc.in_features // dr_n) ** 2, num_classes))

    def forward(self, x):
        # feature extraction 
        x = self.features(x)
        if self.dr:
            # dimensionality reduction using 1×1 convolutions
            x = self.dr_conv(x)
        batch_size = x.size(0)
        channel_num = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_num, feature_size)
        # bilinear pooling
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        # normalization
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5))
        # classification
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = RBP(num_classes=10)
    print(model)
