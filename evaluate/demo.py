#!/usr/bin/env python
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script takes a random video, selects a random summary
% Then, it evaluates the summary and plots the performance compared to the human summaries
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''
import os 
from summe import *
import numpy as np
import random

from vs import lstm, config, h

''' PATHS '''

HOME = '../SumMe'

HOMEDATA = 'GT/';

HOMEVIDEOS = 'videos/';

def evaluateSumMe(c, vs = None):

    included_extenstions = ['webm']
    
    videoList = [fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]

    frame_score = []#len(videoList)
    
    if vs is None:
        
        vs = lstm.vs(c)
    
    for vext in videoList:
            
        v = vext.split('.')[0]

        print('[*] Video Name : ', videoName)

        gt_file = os.path.join(HOME, HOMEDATA, v + '.mat')
    
        gt_data = scipy.io.loadmat(gt_file)
    
        nFrames = gt_data.get('nFrames')

        frame_score = vs.inference(os.path.join(HOME, HOMEVIDEOS, v), 'api', sampling = False, dataset = 'SumMe') > c.ith

        summarys = []

        summarys.append(list(frame_score))
    
    [f_measure, summary_length] = evaluateSummary(summary_selections[0], videoName, HOMEDATA)

    print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))
