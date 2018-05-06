import tensorflow as tf
import data
import network
from lstm import vs
from config import config
import argparse
import h


parser = argparse.ArgumentParser(description = 'Process some integers.')

parser.add_argument('-t', '--train', action = 'store_true', default = False)

parser.add_argument('-i', '--inference', action = 'store_true', default = False)

parser.add_argument('-gpu', action = 'store_true', default = False)

parser.add_argument('-ct', action = 'store_true', default = False) # cross testing

parser.add_argument('-v', '--video', default = '')

parser.add_argument('-m', '--model', default = '')

parser.add_argument('-mfe', type = str, default = 'c3d-sports1M_weights.h5') # model for feature extration

parser.add_argument('-mvs', type = str, default = '') # model for video summarization

parser.add_argument('-dtest', type = str, default = '') # model for video summarization

parser.add_argument('-dtrain', type = str, default = '') # model for video summarization

parser.add_argument('-c', '--checkpoint', default = '')

parser.add_argument('-s', '--step', type = int, default = 2500)

parser.add_argument('-lr', type = float, default = 0.00001)

parser.add_argument('-lrft', type = float, default = 0.9)

parser.add_argument('-lrds', type = int, default = 1500)

parser.add_argument('-th', type = float, default = 0.15)

parser.add_argument('-ts', type = int, default = 10) # test step

parser.add_argument('-np', type = int, default = 1) # np ratio

parser.add_argument('-ss', type = int, default = 2500) # save model per step

parser.add_argument('-fl', type = str, default = 'unit') # feature level [unit, frame]

args = parser.parse_args()

if ((not args.train) and (not args.inference)):

    raise ValueError('--train or --inference')


c = config()

#c.dataset_name = 'SumMe'

c.dtest = args.dtest

c.dtrain = args.dtrain

c.nbatch = 5

c.nhidden = [128, 128]

c.width = 224

c.height = 224

c.size = h.size([224, 224])

c.lr = args.lr

c.threshold = args.th

c.max_nframes = 20000 # original max nframes

c.feature_level = args.fl

c.skip = 5

c.erange = 5

c.arch = 'lstm'

c.net = 'c3d'

c.test_step = args.ts

c.gpu = args.gpu

c.mfe = args.mfe

c.mvs = args.mvs

c.np_ratio = args.np

c.ss = args.ss # save step

c.check()

c.lrds = args.lrds

c.lrft = args.lrft

l = vs(c)

if args.train:

    l.train(args.step, args.checkpoint)

elif args.inference:

    l.inference(args.video, args.model, args.th)






