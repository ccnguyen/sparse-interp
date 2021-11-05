import torch

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)

    pixel_coords -= 0.5
    pixel_coords *= 2.

    temp = pixel_coords[0, :, :, 0].copy()
    pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 1]
    pixel_coords[0, :, :, 1] = temp
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def norm_coord(coords, len=512):
    ''' Convert image from [0, 512] pixel length to [-1, 1] coords'''
    coords = coords / len
    coords -= 0.5
    coords *= 2
    return coords.detach().cpu().numpy()