import torch
import torch.nn as nn
import numpy as np
from utils import norm_coord
from pykdtree.kdtree import KDTree


class TreeMultiQuad(nn.Module):
    '''
    Set up for a
    R G
    G B
    Bayer array'''
    def __init__(self, sz=512, k=4):
        super().__init__()
        self.k = k
        self.sz = sz

    def _find_idx(self, mask):
        '''
        Given a mask, find the idxs that are the closest
        @param mask:
        @return:
        '''
        img = torch.ones((self.sz, self.sz)).cuda()
        holed_img = img * mask
        filled_idx = (holed_img != 0).nonzero()
        unfilled_idx = (holed_img == 0).nonzero()

        # normalize coordinates to [-1, 1]
        filled_idx_n = norm_coord(filled_idx, self.sz)  # size: num_coords, 2
        unfilled_idx_n = norm_coord(unfilled_idx, self.sz)  # size: num_coords, 2

        # define tree
        tree = KDTree(filled_idx_n)
        # for all the idxs you need to fill up, find out the closest filled idxs
        dist, idx = tree.query(unfilled_idx_n, k=self.k)
        # dist = [unfilled_idx num_coords, k]
        # idx = [unfilled_idx num_coords, k]
        # idx is of a flattened img with values from [0, sz * sz]

        idx = idx.astype(np.int32)

        dist = torch.from_numpy(dist).cuda()
        return idx, dist, filled_idx, unfilled_idx

    def _fill(self, holed_img, params):
        idx, dist, filled_idx, unfilled_idx = params
        vals = 0
        for i in range(self.k):
            # find coords of the points which are knn
            idx_select = filled_idx[idx[:, i]]  # size: num_coords, 2

            # add value of those coords, weighted by their distance
            vals += holed_img[idx_select[:, 0], idx_select[:, 1]] * dist[:, i]
        vals /= torch.sum(dist, dim=1)
        holed_img[unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        return holed_img

    def forward(self, coded):
        sz = self.sz

        # red exposure
        mask = torch.zeros((sz, sz)).cuda()
        mask[::2, ::2] = 1
        holed_img = coded[0, 0, :, :] * mask
        params = self._find_idx(mask)
        red = self._fill(holed_img, params)

        # blue exposure
        mask = torch.zeros((sz, sz)).cuda()
        mask[1::2, 1::2] = 1
        holed_img = coded[0, 0, :, :] * mask
        params = self._find_idx(mask)
        blue = self._fill(holed_img, params)

        # green exposure
        mask = torch.zeros((sz, sz)).cuda()
        mask[0::2, 1::2] = 1
        mask[1::2, 0::2] = 1
        holed_img = coded[0, 0, :, :] * mask
        params = self._find_idx(mask)
        green = self._fill(holed_img, params)

        stacked = torch.stack((red, green, blue), dim=0).unsqueeze(0)
        return stacked


class TreeMultiRandom(nn.Module):
    '''
    Use for irregularly interspaced data to be interpolated into full resolution
    '''
    def __init__(self, sz=512, k=4, num_channels=8):
        super().__init__()
        self.k = k # number of neighboring points to search
        self.sz = sz
        self.num_channels = num_channels # num channels to produce from interpolation

        # lookup_table should be (sz, sz) size tensor with an integer value for each
        # pixel, indicating which pixel should be interpolated for which channel from 1-NUM_CHANNELS
        # e.g. if the pixel has value 4, then it will be used to interpolate the 4th channel only
        self.lookup_table = None

    def _find_idx(self, mask):
        img = torch.ones((self.sz, self.sz)).cuda()
        holed_img = img * mask
        filled_idx = (holed_img != 0).nonzero()
        unfilled_idx = (holed_img == 0).nonzero()

        filled_idx_n = norm_coord(filled_idx, self.sz)  # num_coords, 2
        unfilled_idx_n = norm_coord(unfilled_idx, self.sz)  # num_coords, 2

        tree = KDTree(filled_idx_n)
        dist, idx = tree.query(unfilled_idx_n, k=self.k)

        idx = idx.astype(np.int32)

        dist = torch.from_numpy(dist).cuda()
        return idx, dist, filled_idx, unfilled_idx

    def _fill(self, holed_img, params):
        idx, dist, filled_idx, unfilled_idx = params
        vals = 0
        for i in range(self.k):
            # find coords of the points which are knn
            idx_select = filled_idx[idx[:, i]]  # num_coords, 2

            # add value of those coords, weighted by their distance
            vals += holed_img[idx_select[:, 0], idx_select[:, 1]] * dist[:, i]
        vals /= torch.sum(dist, dim=1)
        holed_img[unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        return holed_img

    def forward(self, coded, lookup_table):
        self.lookup_table = lookup_table

        stacked = torch.zeros((self.num_channels, self.sz, self.sz)).cuda()

        for i in range(1, self.num_channels + 1):
            mask = (self.lookup_table == i)
            holed_img = coded[0, 0, :, :] * mask
            params = self._find_idx(mask)
            filled_img = self._fill(holed_img, params)
            stacked[i-1, :, :] = filled_img

        return stacked.unsqueeze(0)