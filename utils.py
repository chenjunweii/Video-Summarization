def restore(checkpoint, net, device):

    f = h5py.File(checkpoint, 'r')
    
    for k in args:

        if k == 'input' or k == 'groundtruth':

            continue

        nd[k] = mx.nd.array(f[k], device)


def save(checkpoint):

    pass




