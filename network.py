import mxnet as mx
from mxnet import nd

def cross_entropy(p, g):

    return -g * nd.log(nd.clip(p, 1e-5, 1)) - (1 - g) * nd.log(nd.clip(1 - p, 1e-5, 1))

def lstm(nhidden, weight, device, mode, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    if mode != 'train':

        dropout = 0
    
    with net.name_scope():

        net.add(mx.gluon.rnn.LSTM(nhidden[0], dropout = dropout))
        
        net.add(mx.gluon.nn.Dense(1, flatten = False))

    return net


def birnn(nhidden, weight, device, mode, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    if mode != 'train':

        dropout = 0
    
    with net.name_scope():

        net.add(mx.gluon.rnn.LSTM(nhidden[0], bidirectional = True, dropout = dropout))

        #net.add(mx.gluon.rnn.LSTM(nhidden[1], bidirectional = True, dropout = dropout))
        
       # net.add(mx.gluon.rnn.(1, layout = 'NTC'))

        net.add(mx.gluon.nn.Dense(1, flatten = False))

        #net.add(mx.gluon.rnn.LSTM(1))
        
        #net.add(mx.gluon.nn.Activation('sigmoid'))

    return net
    
