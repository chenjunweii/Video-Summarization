import io
import os
import sys
import random
import numpy as np

import mxnet
from mxnet import nd, autograd
from wurlitzer import pipes

# custom

from data import data
from video import *
from network import *

class vs(object):
   
    def __init__(self, c):

        print ('[*] Initiating Video Summarization ...')

        self.c = c

        if self.c.gpu:

            self.device = mx.gpu()

        else:

            self.device = mx.cpu()

        self.lr = c.lr

        self.nd = dict()

        stdout = sys.stdout

        sys.stdout = open(os.devnull, 'w')
        
        sys.stdout = stdout

    def build(self):

        if self.mode == 'training':

            print ('[*] Build Training Network ... ')

            if self.c.arch == 'lstm':

                #BiLSTM(self.symbol['input'], self.weights, self.biases, self.symbol['nframes'], self.nhidden)

                self.net = rnn(self.c.nhidden, self.nd, self.device, self.mode)

                #self.node['p'] = self.net(self.symbol['input'])

                #self.loss = cross_entropy(self.node['p'], self.symbol['targets'])
    
    def extract(self, sess, filename):

        symbol = {};

        print ('[*] Extracting Feature From Video => {}'.format(filename))
        
        symbol['raw'] = tf.placeholder(tf.float32, shape = [None, self.height, self.width, 3], name = 'input')

        stdout = sys.stdout

        sys.stdout = open(os.devnull, 'w')
        
        #vggnet = vgg16(symbol['raw'], 'vgg16_weights.npz', sess)
        
        sys.stdout = stdout

        capture = cv2.VideoCapture(filename)

        self.features = None
        
        fidx = -1

        self.max_nframes = 0

        with pipes() as (out, err):
        
            while True:

                ret, frame = capture.read()

                fidx += 1

                #if fidx % self.skip != 0: # skip

                   #continue

                if frame is None or ret is None:

                   break

                resized = cv2.resize(frame, (self.width, self.height)).reshape([1, self.width, self.height, 3])

                self.max_nframes += 1

                feature = sess.run(vggnet.pool5, feed_dict = {symbol['raw'] : resized})
                
                if self.features is None:

                    self.features = feature 

                else:

                    self.features = np.vstack((self.features, feature))
        
        self.features = self.features.reshape([-1, self.max_nframes, 25088])

        self.nframes = [self.max_nframes]

    def train(self, target_step, checkpoint):

        self.mode = 'training'

        print ('[*] Mode : {}'.format('Training'))

        self.c.dataset_name = self.c.dtrain
    
        d = data(self.c)

        d.next()

        dt = None

        if self.c.test_step > 0:

            import copy

            self.ct = copy.deepcopy(self.c)

            print('dataset size : ', d.dataset_size)

            self.ct.dataset_name = self.c.dtest

            self.ct.nbatch = d.dataset_size

            dt = data(self.ct)

            dt.next()

            self.nd['test_input'] = nd.array(dt.data, self.device)
            
            self.nd['test_target'] = nd.array(dt.score, self.device)

        print ('[*] Data Shape : ', d.data.shape)

        print ('[*] Target Shape : ', d.score.shape)

        #self.init(max_nunits)

        self.nd['input'] = nd.array(d.data, self.device)
        
        self.nd['target'] = nd.array(d.score, self.device)

        self.build()

        current_step = 0

        if checkpoint != '':

            print ('[*] Restore From CheckPoint => {}'.format(checkpoint))

            current_step = int(checkpoint.split('/')[-1].split('.')[0]) + 1

            self.net.load_params(checkpoint, ctx = self.device)

        else:

            print('[*] Initialize Parameters ... ')

            self.net.collect_params().initialize(mx.init.Xavier(), ctx = self.device)
           
        print ('[*] Start Training ...')
        
        lr_scheduler = mx.lr_scheduler.FactorScheduler(self.c.lrds, self.c.lrft)   
        
        trainer = mx.gluon.Trainer(self.net.collect_params(), 
                'adam', {'learning_rate': self.c.lr, 
                         'lr_scheduler' : lr_scheduler})
        
        while current_step < target_step + 1:

            with autograd.record():

                prediction = self.net(self.nd['input'])

                np_mask = np.zeros_like(d.score)#np.zeros([self.c.nbatch, d.current_batch_max_nunits, 1])

                effective_unit = 0

                for b in range(self.c.nbatch):

                    n_positive = int(np.sum(d.score[b]))

                    n_negative = int(n_positive * self.c.np_ratio)

                    effective_unit += n_positive

                    effective_unit += n_negative

                    negative_list = list(range(d.current_batch_max_nunits)) 

                    positive_index = np.where(d.score[b] == 1)

                    for p_idx in positive_index[0]:

                        negative_list.remove(p_idx)

                        np_mask[b, p_idx] = 1

                    negative_index = random.sample(negative_list, n_negative)

                    for n_idx in negative_index:

                        np_mask[b, n_idx] = 1

                loss = nd.sum(cross_entropy(prediction, self.nd['target']) * nd.array(np_mask, self.device)) / max(1, effective_unit)
                
                loss.backward()

            trainer.step(self.c.nbatch)

            #print('self.net : ', self.net.collect_params().get('lstm0_l0_i2h_weight').data())

            print ('[*] Step : {}, Loss : {}, LR : {}'.format(current_step, loss.asnumpy()[0], trainer.learning_rate))

            if current_step % self.c.ss == 0 and current_step != 0:

                savepath = 'model/{}.chk'.format(current_step)

                self.net.save_params(savepath) #

                print ('[*] CheckPoint is saved to => {}'.format(savepath))

            if self.c.test_step > 0 and current_step != 0 and current_step % self.c.test_step == 0:

                print('\n[*] ===== Testing ===== \n')

                p = self.net(self.nd['test_input'])

                k = (p > 0.5) #threshold) # key frame

                kmap = np.zeros([self.ct.nbatch, dt.current_batch_max_nunits])

                np_mask = np.zeros_like(kmap)#np.zeros([self.c.nbatch, d.current_batch_max_nunits, 1])

                effective_unit = 0

                for b in range(self.ct.nbatch):

                    n_positive = int(np.sum(dt.score[b]))

                    #print('n positive : ', n_positive)

                    n_negative = int(n_positive * 1)

                    effective_unit += n_positive

                    effective_unit += n_negative

                    negative_list = list(range(dt.current_batch_max_nunits)) 

                    positive_index = np.where(dt.score[b] == 1)

                    for p_idx in positive_index[0]:

                        negative_list.remove(p_idx)

                        np_mask[b, p_idx] = 1

                    negative_index = random.sample(negative_list, n_negative)
                    
                    for n_idx in negative_index:

                        np_mask[b, n_idx] = 1

                for b in range(self.ct.nbatch):

                    kmap[b, : dt.nunits[b]][:] = 1
                
                #`a = float((1 - np.sum(nd.abs(k - self.nd['test_target']).asnumpy().reshape([self.ct.nbatch, -1]) * kmap * np_mask) / np.sum(dt.nunits)))# accuracy
                a = float((1 - np.sum(nd.abs(k - self.nd['test_target']).asnumpy().reshape([self.ct.nbatch, -1]) * kmap * np_mask) / max(effective_unit, 1)))# accuracy

                print('[*] Accuracy : {:.3f} %'.format((a * 100)))

                print('\n[*] =============== \n')
                
                dt.next()
                
                self.nd['test_input'] = nd.array(dt.data, self.device)
                
                self.nd['test_target'] = nd.array(dt.score, self.device)

            d.next()

            self.nd['input'] = nd.array(d.data, self.device)

            self.nd['target'] = nd.array(d.score, self.device)
            
            current_step += 1

    def inference(self, filename, model, threshold):
        
        print ('[*] Mode : {}'.format('Inference'))

        self.mode = 'inference'
        
        if model == '':

            raise ValueError('Model using for inference is not set properly')

        if not os.path.isfile(filename):

            raise ValueError('File => {} is not exist', filename)

       # sampling
       
        unit_feature, unit_id = unit.sampling(filename, self.c.size, self.c.unit_size, self.c.sample_rate, self.c.net, self.c.gpu, self.c.mfe, reuse = False, mode = self.mode)

        self.nd['input'] = nd.array(unit_feature, self.device)
        
        self.build()
        
        self.net.load_params(self.c.mvs)

        score = self.net(self.nd)

        print ('[*] Data Shape => ', self.features.shape)

        generate_video(filename, score, threshold)

