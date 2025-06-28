import torch.nn as nn

def conv1x1(in_channels, out_channels, stride, padding):
		model = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True)
		)
		return model

def conv3x3(in_channels, out_channels, stride, padding):
		model = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True)
		)
		return model

class ResidualBlock(nn.Module):
		def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
				super(ResidualBlock, self).__init__()
				self.downsample = downsample

				if self.downsample:
						self.layer = nn.Sequential(
								conv1x1(in_channels, middle_channels, 2, 0),
								conv3x3(middle_channels, middle_channels, 1, 1),
								conv1x1(middle_channels, out_channels, 1, 0)
						)
						self.downsize = conv1x1(in_channels, out_channels, 2, 0)
				else:
						self.layer = nn.Sequential(
								conv1x1(in_channels, middle_channels, 1, 0),
								conv3x3(middle_channels, middle_channels, 1, 1),
								conv1x1(middle_channels, out_channels, 1, 0)
						)
						self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)

		def forward(self, x):
				if self.downsample:
						out = self.layer(x)
						x = self.downsize(x)
						return out + x
				else:
						out = self.layer(x)
						if x.size() is not out.size():
								x = self.make_equal_channel(x)
						return out + x

class ResNet50_layer4(nn.Module):
		def __init__(self, num_classes=10):
				super(ResNet50_layer4, self).__init__()
				self.layer1 = nn.Sequential(
						nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
						nn.BatchNorm2d(64),
						nn.ReLU(inplace=True),
						nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
				)
				self.layer2 = nn.Sequential(
						ResidualBlock(64, 64, 256, downsample=False),
						ResidualBlock(256, 64, 256, downsample=False),
						ResidualBlock(256, 64, 256, downsample=True),
				)
				self.layer3 = nn.Sequential(
						ResidualBlock(256, 128, 512, downsample=False),
						ResidualBlock(512, 128, 512, downsample=False),
						ResidualBlock(512, 128, 512, downsample=False),
						ResidualBlock(512, 128, 512, downsample=True),
				)
				self.layer4 = nn.Sequential(
						ResidualBlock(512, 256, 1024, downsample=False),
						ResidualBlock(1024, 256, 1024, downsample=False),
						ResidualBlock(1024, 256, 1024, downsample=False),
						ResidualBlock(1024, 256, 1024, downsample=False),
						ResidualBlock(1024, 256, 1024, downsample=False),
						ResidualBlock(1024, 256, 1024, downsample=False),
				)

				self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
				self.fc = nn.Linear(1024, num_classes)

				for m in self.modules():
						if isinstance(m, nn.Linear):
								nn.init.xavier_uniform_(m.weight.data)
						elif isinstance(m, nn.Conv2d):
								nn.init.xavier_uniform_(m.weight.data)

		def forward(self, x):

				out = self.layer1(x)
				out = self.layer2(out)
				out = self.layer3(out)
				out = self.layer4(out)
				out = self.avgpool(out)
				out = out.view(out.size()[0], -1)
				out = self.fc(out)

				return out
