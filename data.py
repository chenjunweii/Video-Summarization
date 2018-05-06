from scipy import io
import tensorflow as tf
import numpy as np

import h5py
import cv2
import os
import random

from TURN import unit

def load_vsumm(dataset_path = 'VSUMM/database', skip = None, c = None, mode = 'unit', ext = '.mpg'):
    
    dataset = 'VSUMM'

    summary_path = 'VSUMM/UserSummary/'

    feature_path = os.path.join('feature', dataset, mode)

    f = open(os.path.join(summary_path, 'videos.txt'), 'r')
    
    line = f.readline()# Header
    
    line = f.readline()# Frist Item
 
    dataset = []

    data = []

    nframes = []

    net = c.net

    print('load vsumm')
    
    if not os.path.isdir(feature_path):
       
        os.makedirs(feature_path)

    while(line):
        
        splitted = line.split('\t')

        video = splitted[0].strip() # video id

        dataset.append(video)

        #nframes.append(int(splitted[-3].replace(',', "")))


        capture = cv2.VideoCapture(os.path.join(dataset_path, video + '.mpg'))

        if skip is None or skip == 0:

            nframes.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        elif skip != 0:

            nframes.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / skip + skip)

        else:

            raise ValueError('Incorrect of Parameter Skip')

        capture.release()
        
        line = f.readline()
        
        if mode == 'unit':

            if not os.path.isfile(os.path.join(feature_path, '{}_US[{}]_SR[{}].h5'.format(video, c.unit_size, c.sample_rate))):

                print('Extracting Unit Level Feature for [ {} ]'.format(os.path.join(dataset_path, video + ext)))

                net = unit.sampling(video + ext, c.size, c.unit_size, c.sample_rate, net, c.gpu, c.mfe, dataset_path, feature_path, reuse = True)

    for i in range(len(dataset)):

        summary = []

        for u in os.listdir(os.path.join(summary_path, dataset[i])):                   
            
            user_summary = [0] * nframes[i]
            
            for s in os.listdir(os.path.join(summary_path, dataset[i], u)):
            
                if '.eps' in s:

                    continue

                user_summary[int(s.strip('.jpeg')[5:])] = 1

            summary.append(user_summary)

        summary = np.asarray(summary).sum(axis = 0)

        data.append(summary)
        
        # data => list [nd, nd, nd ....]

        #print summary.shape

    #print np.asarray(data).shape

    # importtance score, video name in dataset
        
    return data, dataset, dataset_path

def load_summe(dataset_path = 'SumMe/videos', skip = None, mode = 'unit', c = None):

    dataset = 'SumMe'

    feature_path = os.path.join('feature', dataset, mode)

    summary_path = 'SumMe/GT'

    video_name = []

    summary = []

    net = c.net

    #if mode == 'frame':

    if not os.path.isdir(feature_path):
       
        os.makedirs(feature_path)

    for v in os.listdir(dataset_path):

        if '.mp4' not in v:

            continue

        v_name = v.split('.mp4')[0] 
            
        if not os.path.isfile(os.path.join(feature_path, '{}_US[{}]_SR[{}].h5'.format(v_name, c.unit_size, c.sample_rate))):

            print('Extracting Unit Level Feature for [ {} ]'.format(os.path.join(dataset_path, v)))

            net = unit.sampling(v, c.size, c.unit_size, c.sample_rate, net, c.gpu, c.mfe, dataset_path, feature_path, reuse = True)
        
        v_name = v.split('.')[0]

        video_name.append(v_name)

        f = io.loadmat(os.path.join(summary_path, v_name + '.mat'))

        summary.append(f['gt_score'])
        
        #for i in summary[0]:

         #   print(i)
        #raise
        
        #if skip is None or skip == 0:

         #   nframes.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        #elif skip != 0:

         #   nframes.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / skip + skip)

        #else:

         #   raise ValueError('Incorrect of Parameter Skip')

    return summary, video_name, dataset_path


class data(object):

    def __init__(self, c):

        self.c = c

        self.dataset_name = c.dataset_name
        
        if self.dataset_name == "VSUMM":

            self.original_score, self.video_name, self.path = load_vsumm(c = c)

        elif self.dataset_name == 'SumMe':

            self.original_score, self.video_name, self.path = load_summe(c = c)

        else:

            print ('No Dataset is Specified')

            raise

        self.dataset_size = len(self.video_name)

        self.shuffle_counter = self.dataset_size

        self.physical = list(range(self.dataset_size))

        self.virtual = list(range(self.dataset_size))
    
        random.shuffle(self.physical)

        self.skipped_frame_index = [] # for from skipping
    
        self.step = 0

        self.videocapture = []

        self.score = [] # importance Score

        self.nframes = [0] * self.c.nbatch

        self.nunits = [0] * self.c.nbatch

        self.width = None

        self.height = None

        self.batch = []

        self.shuffle_counter = 0

        self.data = None


    def next_keyframe_unit_target(self):

        #print('next Keyframe Unit ')

        #print('current_batch_max_nunits : ', self.current_batch_max_nunits)

        self.score = np.zeros((self.c.nbatch, self.current_batch_max_nunits, 1))
        
        for b in range(self.c.nbatch):
            
            for u in range(self.nunits[b]):

                #print('u :' , u)

                #print('nunit : ', self.nunits[b])
                
                #print('self score : ', self.score.shape)

                start = u * self.c.sample_rate

                end = start + self.c.unit_size - 1

                unit_score = None

                if self.dataset_name == 'SumMe':

                    unit_score = np.mean(self.original_score[self.physical[self.batch[b]]][start : end])

                elif self.dataset_name == 'VSUMM':
                
                    unit_score = np.mean(self.original_score[self.physical[self.batch[b]]][start : end])
                
                if unit_score > self.c.threshold:

                    self.score[b, u] = 1.0

                    #for offset in range(self.erange):

                        #if u + offset < self.nunits[b]:

                        #    self.score[b, u] = 1.0

                        #if u - offset > 0:

                         #   self.score[b, u] = 1.0


    def next_keyframe_skip(self):

        #print self.max_nframes

        print('b : ', self.c.nbatch)

        print('max : ', self.c.max_nframes)
        
        self.score = np.zeros((self.c.nbatch, self.c.max_nframes, 1))

        queue = []

        # find keyframe
            
        # print 'next keyframe skip ...'

        for b in range(self.c.nbatch):

            s = [0] * self.nframes[b]

            q = []

            #print 'nframes : ', self.nframes[b]

            for f in range(self.nframes[b]):

                if self.original_score[self.physical[self.batch[b]]][f] > self.threshold:

                    q.append(f)

            queue.append(q)

        self.skipped_frame_index = []

        # make neighbor frame of keyframe as keyframe

        for b in range(self.c.nbatch):

            #s = [0] * self.nframes[b]

            idx = []

            for f in range(self.nframes[b]):

                if f in queue[b]:

                    m = int(f / self.skip) # map to skipped score

                    print(self.score.shape)

                    print('b : ', b)

                    print('m : ', m)

                    self.score[b, m] = 1.0

                    for offset in range(self.erange):

                        if m + offset < self.nframes[b]:

                            self.score[b, m + offset] = 1.0

                        if m - offset > 0:

                            self.score[b, m - offset] = 1.0
                
                elif f % self.skip != 0: # skip the frame

                    idx.append(f)

            print ('[*] KeyFrame / Frames : {} / {} = {} %'.format(int(np.sum(self.score[b])), self.score[b].shape[0], int(100 * np.sum(self.score[b]) / self.score[b].shape[0])))

            self.skipped_frame_index.append(idx)

    def next_keyframe(self):

        self.score = np.zeros((self.nbatch, self.max_nframes, 1))

        for b in range(self.nbatch):
            
            #s = [0] * self.nframes[b]

            #print 'original_score : ', self.original_score[self.physical[self.batch[b]]].shape

            #print 'sequence length : ', self.nframes[b]
            
            for f in range(self.nframes[b]):

                if self.original_score[self.physical[self.batch[b]]][f] > self.threshold:

                    self.score[b, f] = 1.0;

                    for offset in range(self.erange):

                        if f + offset < self.nframes[b]:

                            self.score[b, f + offset] = 1.0

                        if f - offset > 0:

                            self.score[b, f - offset] = 1.0

    def next(self):

        """

        Forward Function

        """
        
        self.next_batch() # Load Index of Next Batch

        for b in range(self.c.nbatch):



            #if self.skip > 0:

             #   self.nframes[b] = self.original_score[self.physical[self.batch[b]]].shape[0] / self.skip + self.skip

            #elif self.skip == 0:
            
            if self.c.feature_level == 'frame':

                self.nframes[b] = self.original_score[self.physical[self.batch[b]]].shape[0]

            elif self.c.feature_level == 'unit':

                self.nunits[b] = int(self.original_score[self.physical[self.batch[b]]].shape[0] / self.c.sample_rate)

            #else:

             #   raise ValueError ('Incorrect Parameter Skip')
        
        if self.c.feature_level == 'frame':

            if self.c.skip == 0:

                self.next_keyframe()

            elif self.c.skip > 0:

                self.next_keyframe_skip() # find keyframe index

            self.next_feature() # delete skipped frame
        
        elif self.c.feature_level == 'unit':

            self.next_feature()

            return self.current_batch_max_nunits
    
    def next_batch(self):
        
        begin = self.step * self.c.nbatch % self.dataset_size

        end = (self.step + 1) * self.c.nbatch % self.dataset_size

        if (begin + self.c.nbatch) > self.dataset_size:

            self.batch = self.virtual[ begin : ] + self.virtual [ : end ]

        elif (begin + self.c.nbatch) == self.dataset_size:

            self.batch = self.virtual[ begin : ]

        else:

            self.batch = self.virtual[ begin : end ]

        self.step += 1;

        self.shuffle_counter -= self.c.nbatch;

        if self.shuffle_counter <= 0:

            random.shuffle(self.physical);

            self.shuffle_counter = self.dataset_size

    def next_feature(self):

        raw = []
        
        for b in range(self.c.nbatch):

            feature_filepath = os.path.join('feature',
                    self.dataset_name,
                    self.c.feature_level,
                    '{}_US[{}]_SR[{}].h5'.format(self.video_name[self.physical[self.batch[b]]], self.c.unit_size, self.c.sample_rate))
            
            if b == 0:

                pass#print ('[*] Load Feature => {}'.format(feature_filepath))

            else:

                pass#print ('                 => {}'.format(feature_filepath))
            
            H5 = h5py.File(feature_filepath, 'r')

            if self.c.feature_level == 'frame': 

                shape = H5['data'].shape

                if self.skip > 0:

                    raw.append(np.delete(H5['data'], self.skipped_frame_index[b], axis = 0))

                else:

                    raw.append(H5['data'])

            elif self.c.feature_level == 'unit':

                raw.append(H5)

        if self.c.feature_level == 'unit':

            """
            
            Get Maximum unit length of all the batch

            """

            self.current_batch_max_nunits = 0

            self.current_batch_max_nunits = max([len(r) for r in raw]) - 4

            self.data = np.zeros([self.c.nbatch, self.current_batch_max_nunits, self.c.unit_feature_size])

            self.next_keyframe_unit_target()

        else:

            self.data = np.zeros([self.c.nbatch, self.current_batch_max_nframes, raw[b].shape[-1]])
        
        for b in range(self.c.nbatch):

            if self.c.feature_level == 'frame':

                shape = raw[b].shape

                #print 'Shape[0] : ', shape[0]

                if shape[0] != self.c.max_nframes:
                    
                    padding = np.zeros((self.c.max_nframes - shape[0], shape[1]))
                   
                    raw[b] = np.vstack((raw[b], padding))
                    
                    raw[b] = raw[b].reshape((1, self.c.max_nframes, shape[1]))

                else:

                    raw[b] = np.asarray(raw[b]).reshape((1,) + shape)

                self.data[b] = raw[b]

            elif self.c.feature_level == 'unit':

                for u in range(self.current_batch_max_nunits):

                    start = u * self.c.sample_rate

                    end = start + self.c.unit_size - 1

                    #print(list(raw[b].keys()))

                    #print('nframe : ', int(np.asarray(raw[b]['nframes'])))

                    if end < int(np.asarray(raw[b]['nframes'])):

                        try:

                            self.data[b, u][:] = raw[b]['{}_{}'.format(start, end)]

                        except:

                            self.data[b, u][:] = np.zeros([1, self.c.unit_feature_size])
                    else:

                        self.data[b, u][:] = np.zeros([1, self.c.unit_feature_size])

    def next_video(self):

        next_batch()

        frames = []

        nframes = []

        max_nframes = 0

        capture = []
        
        for b in range(nbatch):

            capture.append(cv2.VideoCapture(video_name[physical[b]]))

            if capture[b].isOpened():
            
                nframes[b] = capture[b].get(cv2.cv.CAP_PROP_FRAME_COUNT)
           
            max_nframes = max(max_nframes, nframes[b])
        
        for b in range(nbatch):
 
            for f in range(max_nframes):

                if max_nframes < nframes[b]:

                    ret, frame = capture[b].read()

                    resized = cv2.resize(frame, (width, height))[...,::-1].reshape(1, height, width, 3)

                    if b == 0:
                        
                        frames.append(resized)

                    else:

                        frames[f] = np.vstack((frames[f], resized))

            capture[b].release()

        return max_nframes
