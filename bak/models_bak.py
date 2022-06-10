## TODO: define the convolutional neural network architecture

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# 20,417,004 trainable params

class Net(nn.Module):

    def __init__(self, drop_p=0.5):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
                
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## Block 1
        # 1 input image channel (grayscale)
        # 32 output channels/feature maps
        # 5x5 square convolution kernel
        # input size: batch_size x 1 x 224 x 224
        # output size: batch_size x 32 x 220 x 220 (look formula in docu)
        # (W-F)/S + 1 = (224-5)/1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input size: batch_size x 32 x 220 x 220
        # output size: batch_size x 32 x 110 x 110 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 220/2 = 110
        self.pool1 = nn.MaxPool2d(2,2)
        #self.norm1 = nn.BatchNorm2d(32) # num channels; parameters learned!
        #self.dropout1 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 2
        # 32 input image channels
        # 64 output channels/feature maps
        # 5x5 square convolution kernel
        # input size: batch_size x 32 x 110 x 110
        # output size: batch_size x 64 x 220 x 220 (look formula in docu)
        # (W-F)/S + 1 = (110-5)/1 + 1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        # input size: batch_size x 64 x 106 x 106
        # output size batch_size x 64 x 53 x 53 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 106/2 = 53
        self.pool2 = nn.MaxPool2d(2,2)
        #self.norm2 = nn.BatchNorm2d(64) # num channels; parameters learned!
        #self.dropout2 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 3
        # 64 input image channels
        # 128 output channels/feature maps
        # 3x3 square convolution kernel
        # input size: batch_size x 64 x 53 x 53
        # output size: batch_size x 128 x 53 x 53 (look formula in docu)
        # (W-F)/S + 1 = (53-3)/1 + 1 = 51
        self.conv3 = nn.Conv2d(64, 128, 3)
        # input size: batch_size x 128 x 51 x 51
        # output size: batch_size x 128 x 25 x 25 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 51/2 = 25.5 -> 25
        self.pool3 = nn.MaxPool2d(2,2)
        #self.norm3 = nn.BatchNorm2d(128)
        #self.dropout3 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 4
        # 128 input image channels
        # 256 output channels/feature maps
        # 2x2 square convolution kernel
        # input size: batch_size x 128 x 25 x 25
        # output size: batch_size x 256 x 25 x 25 (look formula in docu)
        # (W-F)/S + 1 = (25-2)/1 + 1 = 24
        self.conv4 = nn.Conv2d(128, 256, 2)
        # input size: batch_size x 256 x 24 x 24
        # output size: batch_size x 256 x 12 x 12 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 24/2 = 12
        # 256 x 12 x 12 = 30976
        self.pool4 = nn.MaxPool2d(2,2)
        #self.norm4 = nn.BatchNorm2d(256)
        #self.dropout4 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 5
        # 256 input image channels
        # 512 output channels/feature maps
        # 1x1 square convolution kernel
        # input size: batch_size x 256 x 12 x 12
        # output size: batch_size x 512 x 12 x 12 (look formula in docu)
        # (W-F)/S + 1 = (12-1)/1 + 1 = 12
        self.conv5 = nn.Conv2d(256, 512, 1)
        # input size: batch_size x 512 x 12 x 12
        # output size: batch_size x 512 x 6 x 6 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 12/2 = 6
        # 512 x 6 x 6 = 18432
        self.pool5 = nn.MaxPool2d(2,2)
        #self.norm5 = nn.BatchNorm2d(512)
        self.dropout5 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 6
        # 512 input image channels
        # 1024 output channels/feature maps
        # 1x1 square convolution kernel
        # input size: batch_size x 512 x 6 x 6
        # output size: batch_size x 1024 x 6 x 6 (look formula in docu)
        # (W-F)/S + 1 = (6-1)/1 + 1 = 6
        self.conv6 = nn.Conv2d(512, 1024, 1)
        # input size: batch_size x 1024 x 6 x 6
        # output size: batch_size x 1024 x 3 x 3 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 6/2 = 3
        # 1024 x 3 x 3 = 9216
        self.pool6 = nn.MaxPool2d(2,2)
        #self.norm6 = nn.BatchNorm2d(1024)
        self.dropout6 = nn.Dropout(p=np.round(drop_p,2))


        # input features: batch_size x 1024 x 3 x 3; 1024 x 3 x 3 = 9216
        self.linear1 = nn.Linear(9216,2000)
        self.dropout7 = nn.Dropout(p=np.round(drop_p,2))
        
        self.linear2 = nn.Linear(2000,500)
        self.dropout8 = nn.Dropout(p=np.round(drop_p,2))
        
        # 68 x 2 keypoints = 136
        self.linear3 = nn.Linear(500,136)
        

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.norm1(x)
        #x = self.dropout1(x)
        #print(x.size())

        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.norm2(x)
        #x = self.dropout2(x)
        #print(x.size())

        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        #x = self.norm3(x)
        #x = self.dropout3(x)
        #print(x.size())

        # Block 4
        x = self.pool4(F.relu(self.conv4(x)))
        #x = self.norm4(x)
        #x = self.dropout4(x)
        #print(x.size())

        # Block 5
        x = self.pool5(F.relu(self.conv5(x)))
        #x = self.norm5(x)
        #x = self.dropout5(x)
        #print(x.size())

        # Block 6
        x = self.pool6(F.relu(self.conv6(x)))
        #x = self.norm6(x)
        x = self.dropout6(x)
        #print(x.size())

        # Flatten: batch_size x 1024 x 3 x 3; 1024 x 3 x 3 -> (batch_size, 9216)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(self.linear1(x))
        x = self.dropout7(x)

        x = F.relu(self.linear2(x))
        x = self.dropout8(x)

        x = self.linear3(x)

        # A modified x, having gone through all the layers of your model, should be returned
        # We can reshape if desired
        # (batch_size, 136) -> (batch_size, 68 (infer), 2)
        # x = x.view(x.size(0),-1,2)
        #print(x.size())
    
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        #nn.init.kaiming_uniform_(m.weight.data)
        nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        nn.init.constant_(m.bias.data, 0)

