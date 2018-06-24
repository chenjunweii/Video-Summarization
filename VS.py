import data
import network
from vs import vs
from config import config
import argparse
import h
import numpy as np

parser = argparse.ArgumentParser(description = 'Process some integers.')

parser.add_argument('-t', '--train', action = 'store_true', default = False)

parser.add_argument('-ot', action = 'store_true', default = False)

parser.add_argument('-i', '--inference', action = 'store_true', default = False)

parser.add_argument('-gpu', action = 'store_true', default = False)

parser.add_argument('-device', type = int, default = 0)

parser.add_argument('-ct', action = 'store_true', default = False) # cross testing

parser.add_argument('-v', '--video', default = '')

parser.add_argument('-m', '--model', default = '')

parser.add_argument('-mfe', type = str, default = 'c3d-sports1M_weights.h5') # model for feature extration

parser.add_argument('-mvs', type = str, default = '') # model for video summarization

parser.add_argument('-dtest', type = str, default = '') # model for video summarization

parser.add_argument('-dtrain', type = str, default = '') # model for video summarization

parser.add_argument('-c', '--checkpoint', default = '')

parser.add_argument('-s', '--step', type = int, default = 2500)

parser.add_argument('-ps', type = int, default = 1000)

parser.add_argument('-lr', type = float, default = 0.00001)

parser.add_argument('-lrft', type = float, default = 0.9)

parser.add_argument('-lrds', type = int, default = 1500)

parser.add_argument('-tth', type = float, default = 0.5) # train

parser.add_argument('-ith', type = float, default = 0.7) # inference

parser.add_argument('-ts', type = int, default = 10) # test step

parser.add_argument('-np', type = int, default = 1) # np ratio

parser.add_argument('-nd', type = int, default = 1) # number of devices

parser.add_argument('-b', type = int, default = 5) # batch size

parser.add_argument('-tb', type = int, default = 10) # test batch size

parser.add_argument('-ss', type = int, default = 500) # save model per step

parser.add_argument('-fl', type = str, default = 'unit') # feature level [unit, frame]

parser.add_argument('-arch', type = str, default = 'lstm') # feature level [unit, frame]

parser.add_argument('-an', type = str, default = 'binary') # annotation method, keyframe or importance score

parser.add_argument('-sm', type = str, default = 'unit') # sampling method, frame or unit

parser.add_argument('-lo', type = str, default = 'TNC') # TNC or NTC

parser.add_argument('-sr', type = int, default = 8) # TNC or NTC

parser.add_argument('-us', type = int, default = 16) # TNC or NTC

args = parser.parse_args()

if ((not args.train) and (not args.inference) and (not args.ot)):

    raise ValueError('--train or --inference' or '-ot')


c = config()

#c.dataset_name = 'SumMe'

c.dtest = args.dtest

c.dtrain = args.dtrain

c.nbatch = args.b

c.device = args.device

c.nhidden = [128]

c.width = 224

c.height = 224

c.size = h.size([224, 224])

c.lr = args.lr

c.tb = args.tb # test batch

c.tth = args.tth # threshold for dataset 

c.ith = args.ith # threshold for prediction

c.max_nframes = 20000 # original max nframes

c.unit_size = args.us

c.sample_rate = args.sr

c.feature_level = args.fl

c.skip = 5

c.erange = 5

c.arch = args.arch

c.net = 'c3d'

c.test_step = args.ts

c.gpu = args.gpu

c.mfe = args.mfe

c.mvs = args.mvs

c.np_ratio = args.np

c.ss = args.ss # save step

c.pstep = args.ps

c.check()

c.lrds = args.lrds

c.lrft = args.lrft

c.checkpoint = args.checkpoint

c.ndevice = args.nd

c.step = args.step

c.an = args.an

c.sm = args.sm

c.lo = args.lo

if args.train:

    l = vs(c)
    
    l.train(c.step, c.checkpoint)

elif args.inference:

    l = vs(c)
    
    l.inference(args.video, )

elif args.ot:

    from optimal import find_optimal

    find_optimal(c, np.arange(0.1, 0.7, 0.1), np.arange(0.1, 0.5, 0.1))





