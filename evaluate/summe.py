'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Package to evaluate and plot summarization results
% on the SumMe dataset
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''

import os
import scipy.io
import warnings
import numpy as np
import matplotlib.pyplot as plt

def evaluateSummary_while_training(summary, user_score, dataset):

    summary = list(summary.reshape([-1]))

    gt_threshold = 0

    if dataset == 'TVSum':
        
        user_score = user_score.T
        
        #summray = summary.T

        gt_threshold = 0
        #nframes = user_score.shape[1];
         
        #nbOfUsers = user_score.shape[0];
    
    #elif dataset == 'SumMe':


    #print('Summary : ', summary[50:100])

    #print('Userscore : ', user_score.mean(axis = 1)[50:100])

    nframes = user_score.shape[0];
         
    try:
        
        nbOfUsers = user_score.shape[1];
    

    except:

        nbOfUsers = 1;

        user_score = user_score.reshape([-1,  1])

    #print('[debug] length of summary : ', len(summary))
    
    #print('[debug] Users : ', nbOfUsers)
    #print('nframes : ', nframes)

    #print('mean of summray : ', np.asarray(summary).mean())

    #print('mean of gt_score : ', np.asarray(user_score).mean())

    if len(summary) < nframes:

        if nframes - len(summary) > 10:

            warnings.warn('Pad selection with %d zeros!' % (nframes - len(summary)))
         
        summary.extend(np.zeros(nframes - len(summary)))

    elif len(summary) > nframes:
        
        if len(summary) - nframes > 10:

            warnings.warn('Pad selection with %d zeros!' % (nframes - len(summary)))

        warnings.warn('Crop selection (%d frames) to GT length' %(len(summary) - nframes))       
        
        summary = summary[0 : nframes];
             
    # Compute pairwise f-measure, summary length and recall


    summary_indicator = np.array(list(map(lambda x: (1 if x > 0 else 0), summary))); # if frame score > 0, idx => 1   
    
    user_intersection = np.zeros((nbOfUsers, 1));
    
    user_union = np.zeros((nbOfUsers, 1));
    
    user_length = np.zeros((nbOfUsers, 1));
    
    for userIdx in range(0, nbOfUsers):

        # gt_indicator
    
        gt_indicator = np.array(list(map(lambda x : (1 if x > 0 else 0), user_score[ : , userIdx]))) # if user_score > 0, idx => 1
         
        # to see how many frame of summary that match (intersection) user choosed frame by summing over the frames
       
        user_intersection[userIdx] = np.sum(gt_indicator * summary_indicator); # sum ((user_score > 0) * (frame_score > 0))

        # to see how many frame that one of summary or user choosed
        
        user_union[userIdx] = sum(np.array(list(map(lambda x : (1 if x > 0 else 0), gt_indicator + summary_indicator))));         
                  
        user_length[userIdx] = sum(gt_indicator) # number of frames that user choosed
    
    recall = user_intersection / user_length

    p = user_intersection / np.sum(summary_indicator);
    
    f_measure = []
    
    for idx in range(0, len(p)):
        
        if p[idx] > 0 or recall[idx] > 0:
            
            f_measure.append(2 * recall[idx] * p[idx] / (recall[idx] + p[idx]))
        
        else:
            
            f_measure.append(0)
     
    nn_f_meas = np.max(f_measure);
     
    f_measure = np.mean(f_measure);
     
    nnz_idx = np.nonzero(summary)
     
    nbNNZ = len(nnz_idx[0])
     
    summary_length = float(nbNNZ) / float(len(summary));
     
    recall = np.mean(recall);
     
    p = np.mean(p);
     
    return f_measure, summary_length



def evaluateSummary(summary_selection, videoName, HOMEDATA):
     
    gt_file = os.path.join(HOMEDATA, videoName + '.mat')
     
    gt_data = scipy.io.loadmat(gt_file)

    # user_score => [number of frames, number of users]
     
    user_score = gt_data.get('user_score')
     
    nFrames = user_score.shape[0];
     
    nbOfUsers = user_score.shape[1];

    print('[*] User Score shape : ', user_score.shape)

    print('[*] Number of User : ', nbOfUsers)
    
     # Check inputs

    if len(summary_selection) < nFrames:

        warnings.warn('Pad selection with %d zeros!' % (nFrames - len(summary_selection)))
         
        summary_selection.extend(np.zeros(nFrames - len(summary_selection)))

    elif len(summary_selection) > nFrames:

        warnings.warn('Crop selection (%d frames) to GT length' %(len(summary_selection) - nFrames))       
        
        summary_selection = summary_selection[0 : nFrames];
             
    # Compute pairwise f-measure, summary length and recall

    summary_indicator = np.array(list(map(lambda x: (1 if x > 0 else 0), summary_selection))); # if frame score > 0, idx => 1   
    
    user_intersection = np.zeros((nbOfUsers, 1));
    
    user_union = np.zeros((nbOfUsers, 1));
    
    user_length = np.zeros((nbOfUsers, 1));
    
    for userIdx in range(0, nbOfUsers):

        # gt_indicator
    
        gt_indicator = np.array(list(map(lambda x : (1 if x > 0 else 0), user_score[ : , userIdx]))) # if user_score > 0, idx => 1
         
        # to see how many frame of summary that match (intersection) user choosed frame by summing over the frames
       
        user_intersection[userIdx] = np.sum(gt_indicator * summary_indicator); # sum ((user_score > 0) * (frame_score > 0))

        # to see how many frame that one of summary or user choosed
        
        user_union[userIdx] = sum(np.array(list(map(lambda x : (1 if x > 0 else 0), gt_indicator + summary_indicator))));         
                  
        user_length[userIdx] = sum(gt_indicator) # number of frames that user choosed
    
    recall = user_intersection / user_length

    p = user_intersection / np.sum(summary_indicator);
    
    f_measure = []
    
    for idx in range(0, len(p)):
        
        if p[idx] > 0 or recall[idx] > 0:
            
            f_measure.append(2 * recall[idx] * p[idx] / (recall[idx] + p[idx]))
        
        else:
            
            f_measure.append(0)
     
    nn_f_meas = np.max(f_measure);
     
    f_measure = np.mean(f_measure);
     
    nnz_idx = np.nonzero(summary_selection)
     
    nbNNZ = len(nnz_idx[0])
     
    summary_length = float(nbNNZ) / float(len(summary_selection));
     
    recall = np.mean(recall);
     
    p = np.mean(p);
     
    return f_measure, summary_length


def plotAllResults(summary_selections,methods,videoName,HOMEDATA):
    '''Evaluates a summary for video videoName and plots the results
      (where HOMEDATA points to the ground truth file) 
      NOTE: This is only a minimal version of the matlab script'''
    
    # Get GT data
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    user_score=gt_data.get('user_score')
    nFrames=user_score.shape[0];
    nbOfUsers=user_score.shape[1];    
    print('in plot')
    ''' Get automated summary score for all methods '''
    automated_fmeasure={};
    automated_length={};
    for methodIdx in range(0,len(methods)):
        summaryIndices=np.sort(np.unique(summary_selections[methodIdx]))
        automated_fmeasure[methodIdx]=np.zeros(len(summaryIndices));
        automated_length[methodIdx]=np.zeros(len(summaryIndices));
        idx=0
        for selIdx in summaryIndices:
            if selIdx>0:
                curSummary=np.array(list(map(lambda x: (1 if x>=selIdx else 0),summary_selections[methodIdx])))    
                f_m, s_l = evaluateSummary(curSummary,videoName,HOMEDATA)
                automated_fmeasure[methodIdx][idx]=f_m
                automated_length[methodIdx][idx]=s_l
                idx=idx+1

    
    ''' Compute human score '''
    human_f_measures=np.zeros(nbOfUsers)
    human_summary_length=np.zeros(nbOfUsers)
    for userIdx in range(0, nbOfUsers):
        human_f_measures[userIdx], human_summary_length[userIdx] = evaluateSummary(user_score[:,userIdx],videoName,HOMEDATA);

    avg_human_f=np.mean(human_f_measures)
    avg_human_len=np.mean(human_summary_length)
    

    ''' Plot results'''
    fig = plt.figure()
    p1=plt.scatter(100*human_summary_length,human_f_measures)
    colors=['r','g','m','c','y']
    for methodIdx in range(0,len(methods)):
        p2=plt.plot(100*automated_length[methodIdx],automated_fmeasure[methodIdx],'-'+colors[methodIdx])
        
    plt.xlabel('summary length[%]')
    plt.ylabel('f-measure')
    plt.title('f-measure for video '+videoName)
    legend=list(methods)    
    legend.extend(['individual humans'])
    plt.legend(legend)
    plt.ylim([0,0.85])
    plt.xlim([0,20])
    plt.plot([5, 5],[0, 1],'--k')
    plt.plot([15.1, 15.1],[ 0, 1],'--k')
    plt.show()
