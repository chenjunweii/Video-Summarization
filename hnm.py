import random
import numpy as np

def lhnm(ps, d, np_ratio):
    
    """
    Local Hard Negative Mining

    ps : positive sample

    """

    np_mask = np.zeros_like(d.score)
    
    p_mask = np.zeros_like(d.score)
    
    n_mask = np.zeros_like(d.score)
    
    effective_unit = 0

    num_positive = 0

    num_negative = 0

    # balanced each in batch

    np_mask[:] = d.score * np_ratio

    p_mask[:] = d.score * np_ratio

    for b in range(d.score.shape[0]):

        num_positive_b = int(np.sum(d.score[b]))

        num_negative_b = num_positive_b * np_ratio

        num_positive += num_positive_b

        num_negative += num_negative_b

        n1, n2 = np.where(d.score[b] == 0)
        
        if num_negative_b > len(n1):

            raise ValueError('\n'
            '[!] Maximum length of units of current batch : {}\n'
            '[!] Current Length : {}\n'
            '[!] Pos : {}\n'
            '[!] Neg = {} * Pos => {}\n'
            '[!] After Removing Postive Sample, Negative Sample : {}'.format(
                d.current_batch_max_nunits,
                d.nunits[b],
                num_positive_b,
                np_ratio,
                num_negative_b,
                len(n1)))

        choosed_n = random.sample(list(n1), num_negative_b)

        for n in n1:

            np_mask[b, n] = 1 # positive + negative
            
            n_mask[b, n] = 1 # only positive, before i set this value to 1, but the loss will not be balanced so 

            # if number of negative sample : nubmer of positive sample = 2 : 1

            # then we weighted positive sample loss more

            # so loss of negative sample : loss of positive sample = 1 : 2

    return [np_mask, p_mask, n_mask], [num_positive, num_negative]
    
def ghnm(ps, d, np_ratio):

    """
    Global Hard Negative Mining

    """
    
    np_mask = np.zeros_like(d.score)
    
    p_mask = np.zeros_like(d.score)
    
    n_mask = np.zeros_like(d.score)
    
    num_positive = int(np.sum(d.score))

    num_negative = int(num_positive * np_ratio)

    num_raw_negative = int(np.product(d.score.shape) - num_positive)

    n1, n2, n3 = np.where(d.score == 0) # find out the x y of negative sample

    np_mask[:] = d.score * np_ratio

    p_mask[:] = d.score * np_ratio

    negative_sample_location_list = list(zip(n1, n2))

    if num_negative > len(negative_sample_location_list):

        raise ValueError('\n'
        '[!] Total units of current batch : {} (Maximum Unit * Batch Size)\n'
        '[!] Total Length : {}\n'
        '[!] Pos : {}\n'
        '[!] Neg = {} * Pos => {}\n'
        '[!] After Removing Postive Sample, Negative Sample : {}'.format(
            np.prod(d.score.shape),
            np.sum(d.nunits),
            num_positive,
            np_ratio,
            num_negative,
            len(negative_sample_location_list)))

    choosed_negative_sample_location_list = random.sample(negative_sample_location_list, num_negative)

    for a1, a2 in choosed_negative_sample_location_list:

        np_mask[a1, a2] = 1

        n_mask[a1, a2] = 1

    return [np_mask.swapaxes(0,1), p_mask.swapaxes(0,1), n_mask.swapaxes(0,1)], [num_positive, num_negative, num_raw_negative]
