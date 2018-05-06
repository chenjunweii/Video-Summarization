import os
import cv2
import h5py
import numpy as np
import mxnet as mx

from TURN.c3d import c3d

import h

width = 224

height = 224

size = h.size([width, height])

gpu = True

model = 'c3d-sports1M_weights.h5'

dataset = 'SumMe'

path = {}

path['VSUMM'] = 'VSUMM/database'

path['SumMe'] = 'SumMe/videos'


videos = os.listdir(path[dataset])

fe = c3d('extract', 1, size, gpu, model)                                                                                                                                                                                   
for v in videos:

    if '.mp4' in v or '.mpg' in v:

        h5_path = os.path.join('extract', dataset, v.split('.')[0] + '.h5')

        if os.path.isfile(h5_path):

            continue

        H5 = h5py.File(os.path.join('extract', dataset, v.split(".")[0] + '.h5'), 'w')

        capture = cv2.VideoCapture(os.path.join(path[dataset], v))

        nframes = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        stack = None

        f = 1

        while True:

        #for i in xrange(5):

            print ("{} => Frame {} / {}".format(videos.index(v), f, nframes))

            ret, frame = capture.read()

            #frame = frame - np.array([104, 117, 123])

            if ret == True:

                print(frame.shape)

                print(width)

                print(height)

                #resized = cv2.resize(frame, (120, 120))

                resized = cv2.resize(frame, (width, height)).reshape([1, height, width, 3]) # Make it RGB

                extracted = fe.extract(resized)

                if stack is None:

                    stack = extracted

                else:

                    stack = np.vstack((stack, extracted))

            else:

                break

            f += 1

        H5['data'] = stack

        H5.close()

