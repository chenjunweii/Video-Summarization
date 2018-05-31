import io
import os
import sys
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mxnet
from mxnet import nd, autograd
from wurlitzer import pipes

# custom

try:

    from data import data
    from video import *
    from network import *
    from evaluate import summe
    from hnm import ghnm, lhnm

except:
    
    from .data import data
    from .video import *
    from .network import *
    from .TURN import unit
    from .evaluate import summe
    from .hnm import ghnm, lhnm

class vs(object):
   
    def __init__(self, c):

        print ('[*] Initializing Video Summarization ...')

        self.c = c

        if self.c.gpu:

            self.device = mx.gpu(self.c.device)

        else:

            self.device = mx.cpu()

        self.lr = c.lr

        self.nd = dict()

        stdout = sys.stdout

        sys.stdout = open(os.devnull, 'w')
        
        sys.stdout = stdout

    def build(self):

            #print ('[*] Build Training Network ... ')

        if self.c.arch == 'lstm':

            #BiLSTM(self.symbol['input'], self.weights, self.biases, self.symbol['nframes'], self.nhidden)

            self.net = rnn(self.c.nhidden, self.nd, self.device, self.mode)

        elif self.c.arch == 'bilstm':
            
            self.net = birnn(self.c.nhidden, self.nd, self.device, self.mode)
            
            #self.testnet = rnn(self.c.nhidden, self.nd, self.device, 'inference')

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

        self.mode = 'train'; self.c.mode = 'train';

        print ('[*] Mode : {}'.format('Training'))

        self.c.dataset_name = self.c.dtrain
    
        d = data(self.c)

        d.next()

        dt = None

        if self.c.test_step > 0:

            import copy

            self.ct = copy.deepcopy(self.c)

            self.ct.dataset_name = self.c.dtest

            self.ct.nbatch = self.c.tb

            self.ct.mode = 'test'

            dt = data(self.ct)

            dt.next()

            #dt.score = dt.score.reshape([self.ct.nbatch, -1])

            self.nd['test_input'] = nd.array(dt.data, self.device)
            
            #self.nd['test_target'] = nd.array(dt.score, self.device)

        print ('[*] Data Shape : ', d.data.shape)

        print ('[*] Target Shape : ', d.score.shape)

        #self.init(max_nunits)

        self.nd['input'] = nd.array(d.data, self.device)
        
        self.nd['target'] = nd.array(d.score, self.device)

        self.build()

        current_step = 0

        if checkpoint != '':

            print ('[*] Restore From CheckPoint => {}'.format(checkpoint))

            self.net.load_params(checkpoint, ctx = self.device)

            try:

                current_step = int(checkpoint.split('/')[-1].split('.')[0]) + 1

            except:

                current_step = 0
            
            #self.testnet(checkpoint, ctx = self.device)

            #self.testnet.collect_params() = self.net.collect_params()

        else:

            print('[*] Initialize Parameters ... ')

            self.net.collect_params().initialize(mx.init.Xavier(), ctx = self.device)
           
        print ('[*] Start Training ...')
        
        lr_scheduler = mx.lr_scheduler.FactorScheduler(self.c.lrds, self.c.lrft)   
        
        trainer = mx.gluon.Trainer(self.net.collect_params(), 
                'nadam', {'learning_rate': self.c.lr, 
                         'lr_scheduler' : lr_scheduler,
                         'clip_gradient': 1})
        
        history = dict()

        history['acc'] = dict(); history['loss'] = dict(); history['eval'] = dict()

        history['acc']['step'] = [0]; history['acc']['raw'] = [0]; history['acc']['bal'] = [0];

        history['eval']['step'] = [0]; history['eval']['fscore'] = [0]; history['eval']['length'] = [0];
        
        while current_step < target_step + 1:

            with autograd.record():

                prediction = self.net(self.nd['input'])

                prediction_keyframe = prediction > self.c.ith

                [np_mask, p_mask, n_mask], [n_pos, n_neg, _] = ghnm(prediction_keyframe, d, self.c.np_ratio)
                
                loss = nd.sum(cross_entropy(prediction, self.nd['target']) * nd.array(np_mask, self.device)) / (n_pos + n_neg)
                
                #loss = nd.mean(cross_entropy(prediction, self.nd['target']))# * nd.array(np_mask, self.device)) / (n_positive + n_negative)
                
                loss.backward()

            trainer.step(self.c.nbatch)

            #print('self.net : ', self.net.collect_params().get('lstm0_l0_i2h_weight').data())

            print ('[*] Step : {}, Loss : {:.3f}, LR : {:}, Pos : {}, Neg : {}'.format(current_step, loss.asnumpy()[0], trainer.learning_rate, n_pos, n_neg))

            if current_step % self.c.ss == 0 and current_step != 0:

                savepath = 'model/{}_dtrain[{}]_dtest[{}]_racc[{:.2f}]_bacc[{:.2f}]_tth[{}]_ith[{}]_np[{}]_arch[{}].chk'.format(current_step,
                        self.c.dtrain,
                        self.c.dtest,
                        history['acc']['raw'][-1] * 100,
                        history['acc']['bal'][-1] * 100,
                        self.c.tth,
                        self.c.ith,
                        self.c.np_ratio,
                        self.c.arch)

                self.net.save_params(savepath) #
                
                print ('[*] CheckPoint is saved to => {}'.format(savepath))

            if self.c.test_step > 0 and current_step != 0 and current_step % self.c.test_step == 0:

                print('\n[*] ===== Testing ===== \n')

                p = self.net(self.nd['test_input']).asnumpy()#.reshape([self.ct.nbatch, -1])

                pk = (p >= self.c.ith) #threshold) # positive key frame
                
                [np_mask, p_mask, n_mask], [n_pos, n_neg, n_raw_neg] = ghnm(pk, dt, 1)
                
                eval_fscore = np.zeros([self.ct.nbatch])

                eval_length = np.zeros([self.ct.nbatch])

                for b in range(self.ct.nbatch):

                    if self.ct.dataset_name == 'SumMe' or self.ct.dataset_name == 'TVSum':

                        eval_fscore[b], eval_length[b] = summe.evaluateSummary_while_training(
                                #self.unsampling(dt.score[b], dt.nunits[b]), # > self.c.ith,
                                #dt.user_score[b],
                                self.unsampling(p[b], dt.nunits[b]) > self.c.ith,
                                #dt.original_score[b],
                                dt.user_score[b],
                                #np.mean(dt.original_user_score[b], axis = 1),
                                #dt.original_score[b],
                                #dt.score[b],
                                #dt.score[b],
                                #np.array([0,0,1,1,1]),
                                #np.array([1,1,1,1,1]),
                                self.ct.dataset_name)

                    elif self.ct.dataset_name == 'TVSum':

                        eval_fscore[b], eval_length[b] = tvsum.evaluateSummary()
                """ 
                print('[*] =========================== ')

                print('[*] ith testing\n')

                for i in range(8):

                    print('[*] ------------------------ ')

                    print('[*] tth = {}'.format(self.c.tth))

                    print('[*] ith = {}'.format(i * 0.1))
                    
                    print('[*] Average F-Score : {}'.format(np.mean(eval_plot_fscore[i])))
                    
                    print('[*] Average Summary Length : {}'.format(np.mean(eval_plot_length[i])))

                print('\n[*] =========================== ')
                """
                raw_a = float(1 - np.mean(np.abs(pk - dt.score)))
                
                a = float((1 - np.sum(np.abs(pk - dt.score) * np_mask) / (n_pos + n_neg)))

                print('[*] Raw Accuracy : {:.3f} %'.format(raw_a * 100))
                
                print('[*] Raw Positive / Negative : {} / {}'.format(n_pos, n_raw_neg))
                
                print('[*] Balanced Accuracy : {:.3f} %'.format((a * 100)))
                
                print('[*] Balanced Positive / Negative : {} / {}'.format(n_pos, n_neg))
                
                print('[*] Average F-Score : {:.3f}'.format((eval_fscore.mean())))
                
                print('[*] Average Summray Length : {}'.format(eval_length.mean()))

                history['acc']['step'].append(current_step); history['acc']['raw'].append(raw_a); history['acc']['bal'].append(a)
                
                history['eval']['step'].append(current_step); history['eval']['fscore'].append(eval_fscore.mean()); history['eval']['length'].append(eval_length.mean())
                print('\n[*] =============== \n')
                
                dt.next()
                
                self.nd['test_input'] = nd.array(dt.data, self.device)
                
                self.nd['test_target'] = nd.array(dt.score, self.device)
            
            if current_step % self.c.pstep  == 0 and current_step != 0:
            
                self.plot(current_step, 'Raw Accuracy', history['acc']['step'], history['acc']['raw'])

                self.plot(current_step, 'Balanced Accuracy', history['acc']['step'], history['acc']['bal'])
                
                self.plot(current_step, 'F-Score', history['eval']['step'], history['eval']['fscore'])
                
                self.plot(current_step, 'Summary Length', history['eval']['step'], history['eval']['length'])
            
            d.next()

            self.nd['input'] = nd.array(d.data, self.device)

            self.nd['target'] = nd.array(d.score, self.device)
            
            current_step += 1
        
        self.plot(current_step, 'Raw Accuracy', history['acc']['step'], history['acc']['raw'])

        self.plot(current_step, 'Balanced Accuracy', history['acc']['step'], history['acc']['bal'])

        self.plot(current_step, 'F-Score', history['eval']['step'], history['eval']['fscore'])
        
        self.plot(current_step, 'Summary Length', history['eval']['step'], history['eval']['length'])
        
        #print('[*] Last Model is save to => {}'.format('last.chk'))

        #self.net.save_params('last.chk') #
        
        return history

    def plot(self, current_step, mode, step, accuracy):

        plt.plot(step, accuracy)

        plt.figtext(0.15, 0.62, 'Train Set : {}\n'
                'Testing Set : {}\n' 
                'Train Batch Size : {}\n'
                'Total Step : {}\n'
                'Train Threshold : {}\n'
                'Inference Threshold : {}\n'
                'Negative / Positive : {}\n'
                'Architecture : {}'.format(self.c.dtrain, self.c.dtest, self.c.nbatch, current_step, self.c.tth, self.c.ith, self.c.np_ratio, self.c.arch),
                bbox = {'facecolor' : 'blue', 'alpha' : 0.1, 'pad' : 10})

        plt.xlabel('Step')

        plt.ylabel(mode)
        
        plt.ylim([0, 1])

        plt.savefig('train[{}]_'
        'test[{}]_'
        'batchsize[{}]_'
        'step[{}]_'
        'tth[{}]_'
        'ith[{}]_'
        'npr[{}]_'
        'arch[{}]_{}.png'.format(self.c.dtrain, self.c.dtest, self.c.nbatch, current_step, self.c.tth, self.c.ith, self.c.np_ratio, self.c.arch, mode))

        plt.clf()

    def inference(self, filename, infmode = 'generate', sampling = True, dataset = '', reuse = False):
        
        print ('[*] Mode : {}'.format('Inference'))

        if not os.path.isfile(filename) and dataset == '':

            raise ValueError('File => {} is not exist', filename)

       # sampling

        unit_feature = None

        unit_id = None

        if sampling:
       
            unit_feature, unit_id = unit.sampling(filename, 
                    self.c.size,
                    self.c.unit_size,
                    self.c.sample_rate,
                    self.c.net,
                    self.c.gpu,
                    self.c.mfe,
                    reuse = False,
                    mode = self.mode)

        else:

            import h5py

            filename_without_path_ext = filename.split('.')[0].split('/')[-1]

            print('path : ',
                    os.path.join('vs', 'feature', dataset, 'unit', '{}_US[{}]_SR[{}].h5'.format(filename_without_path_ext,
                self.c.unit_size,
                self.c.sample_rate)))
            
            h5 = h5py.File(os.path.join('vs', 'feature', dataset, 'unit', '{}_US[{}]_SR[{}].h5'.format(filename_without_path_ext,
                self.c.unit_size,
                self.c.sample_rate)))
            
            nframs = int(np.array(h5['nframes']))

            fidx = 0

            nunits = len(list(h5.keys())) - 5

            unit_feature = np.zeros([1, nunits, self.c.unit_feature_size])
                
            for uidx in range(nunits):

                fstart = uidx * self.c.sample_rate

                fend = fstart + self.c.unit_size - 1

                unit_feature[0, uidx][:] = np.array(h5['{}_{}'.format(fstart, fend)])
            
        self.nd['input'] = nd.array(unit_feature, self.device)
        
        self.build()
        
        self.net.load_params(self.c.mvs, self.device)

        score = self.net(self.nd['input']).asnumpy().reshape([-1])

        if infmode == 'generate':

            generate_video(filename, score, threshold)
       
        elif infmode == 'api':

            return self.unsampling(score)

    def unsampling(self, unit_score, nunits = None):
        
        frame_score = None

        effective_number_unit = unit_score.shape[0] if nunits is None else nunits

        frame_score = np.zeros([(effective_number_unit + 1) * self.c.sample_rate])

        for uidx in range(effective_number_unit):

            start = uidx * self.c.sample_rate

            end = start + self.c.unit_size - 1

            frame_score[ start : end ] += unit_score[uidx]
            #frame_score[ start : end ] = unit_score[uidx]

        frame_score[self.c.sample_rate : - self.c.sample_rate] /= 2
        
        return frame_score
