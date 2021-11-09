import torch.nn as nn
from interpolate import TreeMultiQuad, TreeMultiRandom
import torch
from debayer import Debayer3x3


class TreeModel(nn.Module):
    ''' With input of a single channel 1xHxW and a lookup table for
    which pixels should be interpolated to which channel,
    provide a full-resolution, multi-channel output using
    KDTree to interpolate from closest points.
    '''
    def __init__(self, coded_type='irregular', sz=512, num_channels=6):
        ''' coded type can be quad (e.g. Bayer) or irregular
        Inputs:
        coded_type: str that is either 'irregular' or 'quad'
        sz: int of how large H==W is (assuming square image
        '''
        super().__init__()
        self.coded_type = coded_type
        if coded_type == 'quad':
            # regularly spaced points
            self.tree = TreeMultiQuad(sz=sz)
        else:
            # irregular spaced points
            self.tree = TreeMultiRandom(sz=sz, num_channels=num_channels)

    def forward(self, coded, lookup_channels=None):
        ''' Coded is the single channel image we want to stack into multiple channels '''
        if self.coded_type == 'irregular':
            return self.tree(coded, lookup_channels)
        else:
            return self.tree(coded)


class InterpModel(nn.Module):
    ''' Can use a bilinear interpolation module for comparison'''
    def __init__(self):
        super().__init__()
        self.interp = Debayer3x3().cuda()

    def forward(self, coded):
        return self.interp(coded)


if __name__ == '__main__':
    device = 'cuda:0'
    tree = TreeModel(coded_type='irregular', sz=512, num_channels=6)

    # give each pixel a channel assignment
    # i.e. each pixel is expected to belong to a different channel
    lookup_channels = torch.randint(low=0, high=6, size=(512, 512), device=device)

    # get single-channel coded image
    coded = torch.rand(size=(1, 1, 512, 512), device=device)

    interpolated = tree(coded, lookup_channels)
    print(f'Interpolated shape: {interpolated.shape}')
