import threading
import queue
import copy
import time
import matplotlib.pyplot as plt
import numpy as np

from vs import vs

def train(c, ith, tth, fscore, length, device_available, lock, done, total):

    c.device = device_available.get()
    
    c.ith = ith

    c.tth = tth

    c.thread = c.device

    vsi = vs(c)

    print('[*] Device {} is training'.format(c.device))

    h = vsi.train(c.step, c.checkpoint)

    if ith not in fscore.keys():
        fscore[ith] = dict()
        length[ith] = dict()
    if tth not in fscore[ith].keys():
        fscore[ith][tth] = dict()
        length[ith][tth] = dict()

    lock.acquire()

    fscore[ith][tth] = h['eval']['fscore'][-1]

    length[ith][tth] = h['eval']['length'][-1]

    device_available.put(c.device)

    done += 1

    lock.release()

    print('[*] Training of Device {} is done'.format(c.device))

    print('[*] Progress {:2f} %'.format(done[0] / total * 100)) 

    return

def find_optimal(c, ith, tth):

    fscore = dict()#np.zeros([len(ith), len(tth)])
    
    length = dict()#np.zeros([len(ith), len(tth)])

    thread = dict()

    device_available = queue.Queue()

    done = np.array([0])

    total = ith.shape[0] * tth.shape[0]

    lock = threading.Lock()
    
    for d in range(c.ndevice):

        device_available.put(d)

    for i in ith:

        for t in tth:

            while device_available.empty():

                time.sleep(1)
            
            tc = copy.deepcopy(c)

            thread[tc.device] = threading.Thread(target = train, args = (tc, i, t, fscore, length, device_available, lock, done, total))

            thread[tc.device].start()

    for d in range(c.ndevice):

        thread[d].join()

    fscore_nd = np.zeros([ith.shape[0], tth.shape[0]])
    
    length_nd = np.zeros([ith.shape[0], tth.shape[0]])

    for ii in range(ith):

        for it in range(tth):

            fscore_nd[ii, it] = fscore[ith[ii]][tth[it]]

            length_nd[ii, it] = length[ith[ii]][tth[it]]


    for ii in range(ith):

        plot.plot(tth, fscore_nd[ii])


    plt.xlabel('ith')

    plt.ylabel('fscore')
    
    plt.ylim([0, 1])

    prefix = 'train[{}]_'
    'test[{}]_'
    'batchsize[{}]_'
    'step[{}]_'
    'npr[{}]_'
    'arch[{}]_{}'.format(self.c.dtrain, self.c.dtest, self.c.nbatch, current_step, self.c.np_ratio, self.c.arch, mode)

    plt.savefig('{}_{}.png'.format(prefix, 'fscore')

    plt.clf()
    
    for ii in range(ith):

        plot.plot(tth, length_nd[ii])

    plt.xlabel('ith')

    plt.ylabel('length')
    
    plt.ylim([0, 1])

    plt.savefig('{}_{}.png'.format(prefix, 'length')



