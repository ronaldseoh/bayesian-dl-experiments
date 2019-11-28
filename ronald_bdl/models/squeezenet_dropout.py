import torch
import torch.nn as nn
import torch.nn.init as init

from .dropout_custom import create_dropout_layer


class FireDropout(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, **kwargs):
        super(Fire, self).__init__()

        self.inplanes = inplanes
        
        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)        

        # Additional dropout layer
        self.squeeze_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type,),
        
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)

        # Additional dropout layer
        self.expand1x1_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type,),
        
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        
        # Additional dropout layer
        self.expand3x3_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type,),
        
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze_dropout(self.squeeze(x)))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1_dropout(self.expand1x1(x))),
            self.expand3x3_activation(self.expand3x3_dropout(self.expand3x3(x)))
        ], 1)


class SqueezeNetDropout(nn.Module):

    def __init__(self, num_classes=1000, **kwargs):
        super(SqueezeNetDropout, self).__init__()

        self.num_classes = num_classes
        
        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'
            
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            # Additional dropout layer added here
            create_dropout_layer(
                self.dropout_rate, -1, self.dropout_type,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireDropout(64, 16, 64, 64, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            FireDropout(128, 16, 64, 64, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireDropout(128, 32, 128, 128, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            FireDropout(256, 32, 128, 128, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireDropout(256, 48, 192, 192, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            FireDropout(384, 48, 192, 192, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            FireDropout(384, 64, 256, 256, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
            FireDropout(512, 64, 256, 256, dropout_rate=dropout_rate, dropout_type=self.dropout_type),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        
        self.classifier = nn.Sequential(
            # SqueezeNet (and many other known CNNs) originally 
            # do have dropout layers here, right before the final output
            # Replace it with our dropout layer creator
            create_dropout_layer(
                self.dropout_rate, -1, self.dropout_type,),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
