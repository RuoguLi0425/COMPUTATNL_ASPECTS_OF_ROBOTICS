import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=(128 + 256), out_channels=128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=(64 + 128), out_channels=64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=(32 + 64), out_channels=32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=(16 + 32), out_channels=16, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=1)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO

        # left-side
        X1 = F.relu(self.conv1(x))  ## activation
        x = F.max_pool2d(X1, 2)  ## and pooling

        X2 = F.relu(self.conv2(x))
        x = F.max_pool2d(X2, (2, 2))

        X3 = F.relu(self.conv3(x))
        x = F.max_pool2d(X3, (2, 2))

        X4 = F.relu(self.conv4(x))
        x = F.max_pool2d(X4, (2, 2))

        # right-side
        X5 = F.interpolate(F.relu(self.conv5(x)), scale_factor=2)
        x = torch.cat((X4, X5), dim=1)

        X6 = F.interpolate(F.relu(self.conv6(x)), scale_factor=2)
        x = torch.cat((X3, X6), dim=1)

        X7 = F.interpolate(F.relu(self.conv7(x)), scale_factor=2)
        x = torch.cat((X2, X7), dim=1)

        X8 = F.interpolate(F.relu(self.conv8(x)), scale_factor=2)
        x = torch.cat((X1, X8), dim=1)

        # end-segment
        x = F.relu(self.conv9(x))
        output = F.relu(self.conv10(x))

        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)