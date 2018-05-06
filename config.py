class config(object):

    nbatch = None

    width = None

    height = None

    nhidden = None
    
    dataset_name = ''

    mode = ''

    lr = 0.0001

    erange = 5

    threshold = 0.3

    size = None

    unit_size = 16

    unit_feature_size = 4096

    sample_rate = 8

    dirs = dict()

    arch = ''

    net = ''

    skip = 0

    max_nframes = 0;

    feature_level = ''

    checkpoint = ''

    def check(self):

        if self.skip > 0:

           self.max_nframes = int(self.max_nframes / self.skip + self.skip)

        if self.nbatch is None or self.width is None and self.height is None and self.nhidden is None:

            raise ValueError('[!] Config is not set properly')


