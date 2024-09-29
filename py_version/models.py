# model
import torch
import torch.nn as nn
import torch.nn.functional as F

#Unet
# Model architecture
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = F.interpolate(d4, size=e3.size(2), mode='linear', align_corners=True)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out
    




# ResUnet
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        # Upsampling
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        return x




# Define Two 3×3 Convolution Layers
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_convolution = nn.Sequential(
            # First 3×3 convolutional layer
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            # Second 3×3 convolutional layer
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )

    def forward(self, x: torch.tensor):
        # Apply the two convolution layers and activations
        return self.double_convolution(x)


# Define Down-sample
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.tensor):
        return self.pool(x)


# Define Up-sample
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Up-convolution
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.tensor):
        return self.up(x)


# Define the Unet model
class ResUNet(nn.Module):
    # Initialize the layers
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Double convolution layers for the contracting path
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.down_conv = nn.ModuleList(
            [
                DoubleConvolution(i, j)
                for i, j in [(64, 64), (64, 128), (128, 256), (256, 512)]
            ]
        )
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        # The two convolution layers at the lowest resolution
        self.middle_conv = DoubleConvolution(512, 1024)
        # Up sampling layers for the expansive path
        self.up_sample = nn.ModuleList(
            [
                UpSample(i, j)
                for i, j in [(1024, 512), (512, 256), (256, 128), (128, 64)]
            ]
        )
        # Double convolution layers for the expansive path.
        self.up_conv = nn.ModuleList(
            [
                DoubleConvolution(i, j)
                for i, j in [(512, 512), (256, 256), (128, 128), (64, 64)]
            ]
        )
        # Final 1×1 convolution layer to produce the output
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        identity = x
        skip_connection = []
        x = self.first_conv(x)
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            skip_connection.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            skip_tensor = skip_connection.pop()
            if x.shape != skip_tensor.shape:
                x = self._crop_or_pad(x, skip_tensor.shape)
            x = x + skip_tensor
            x = self.up_conv[i](x)
        x = self.final_conv(x)
        x = x + identity
        return x

    def _crop_or_pad(self, x: torch.Tensor, target_shape: torch.Size):
        _, _, target_length = target_shape
        _, _, length = x.shape
        if length > target_length:
            x = x[:, :, :target_length]
        elif length < target_length:
            pad_amount = target_length - length
            x = F.pad(x, (0, pad_amount))
        return x
    


class SimpleCNN(nn.Module):
    def __init__(self,in_channels , out_channels):
        super(SimpleCNN, self ).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=2)
        self.batchnorm0 = nn.BatchNorm1d(num_features=2)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool1d(kernel_size=2)
        
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc = nn.Linear(in_features=11, out_features=100)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        

        x = self.fc(x)
        return x