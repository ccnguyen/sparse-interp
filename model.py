import torch.nn as nn
from interpolate import TreeMultiQuad, TreeMultiRandom


class TreeModel(nn.Module):
    ''' With input of a single channel 1xHxW and a lookup table for
    which pixels should be interpolated to which channel,
    provide a full-resolution, multi-channel output using
    KDTree to interpolate from closest points.
    '''
    def __init__(self, shutter, shutter_type='irregular', sz=512):
        ''' Shutter type can be quad (e.g. Bayer) or irregular'''
        super().__init__()
        self.shutter = shutter
        self.shutter_type = shutter_type
        if shutter_type == 'quad':
            # regularly spaced points
            self.tree = TreeMultiQuad(sz=sz)
        else:
            # irregular spaced points
            self.tree = TreeMultiRandom(sz=sz)

    def forward(self, coded):
        ''' Coded is the single channel image we want to stack into multiple channels '''
        if self.shutter_type == 'irregular':
            return self.tree(coded, self.shutter.getLength())
        else:
            return self.tree(coded)
