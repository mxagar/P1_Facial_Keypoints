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
        self.pool = nn.MaxPool2d(2,2)
        #self.norm1 = nn.BatchNorm2d(32) # num channels; parameters learned!
        #self.dropout1 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 2
        # 32 input image channels
        # 64 output channels/feature maps
        # 5x5 square convolution kernel
        # input size: batch_size x 32 x 110 x 110
        # output size: batch_size x 32 x 220 x 220 (look formula in docu)
        # (W-F)/S + 1 = (110-3)/1 + 1 = 108
        self.conv2 = nn.Conv2d(32, 32, 3)
        # input size: batch_size x 32 x 108 x 108
        # output size batch_size x 32 x 54 x 54 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 108/2 = 54
        #self.pool2 = nn.MaxPool2d(2,2)
        #self.norm2 = nn.BatchNorm2d(32) # num channels; parameters learned!
        #self.dropout2 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 3
        # 64 input image channels
        # 128 output channels/feature maps
        # 3x3 square convolution kernel
        # input size: batch_size x 32 x 54 x 54
        # output size: batch_size x 64 x 52 x 52 (look formula in docu)
        # (W-F)/S + 1 = (54-3)/1 + 1 = 52
        self.conv3 = nn.Conv2d(32, 64, 3)
        # input size: batch_size x 64 x 52 x 52
        # output size: batch_size x 64 x 26 x 26 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 52/2 = 26
        #self.pool3 = nn.MaxPool2d(2,2)
        #self.norm3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p=np.round(drop_p,2))

        ## Block 4
        # 128 input image channels
        # 256 output channels/feature maps
        # 2x2 square convolution kernel
        # input size: batch_size x 64 x 26 x 26
        # output size: batch_size x 128 x 24 x 24 (look formula in docu)
        # (W-F)/S + 1 = (26-3)/1 + 1 = 24
        self.conv4 = nn.Conv2d(64, 64, 3)
        # input size: batch_size x 64 x 24 x 24
        # output size: batch_size x 64 x 12 x 12 (look formula in docu)
        # kernel_size=2, stride=2 -> W = W/2 = 24/2 = 12
        # 64 x 12 x 12 = 9216
        #self.pool4 = nn.MaxPool2d(2,2)
        #self.norm4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p=np.round(drop_p,2))

        # input features: batch_size x 64 x 12 x 12; batch_size x 9216
        self.linear1 = nn.Linear(9216,1000)
        self.dropout5 = nn.Dropout(p=np.round(drop_p,2))
        
        # 68 x 2 keypoints = 136
        self.linear2 = nn.Linear(1000,136)
        

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.norm1(x)
        #x = self.dropout1(x)
        #print(x.size())

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.norm2(x)
        #x = self.dropout2(x)
        #print(x.size())

        # Block 3
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.norm3(x)
        x = self.dropout3(x)
        #print(x.size())

        # Block 4
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.norm4(x)
        x = self.dropout4(x)
        #print(x.size())

        # Flatten: batch_size x 64 x 12 x 12; 64 x 12 x 12 -> (batch_size, 9216)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(self.linear1(x))
        x = self.dropout5(x)

        x = self.linear2(x)

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

def get_num_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params