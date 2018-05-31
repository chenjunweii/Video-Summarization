import mxnet as mx
from mxnet import nd

def cross_entropy(p, g):

    return -g * nd.log(nd.clip(p, 1e-5, 1)) - (1 - g) * nd.log(nd.clip(1 - p, 1e-5, 1))

def rnn(nhidden, weight, device, mode, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    if mode != 'train':

        dropout = 0
    
    with net.name_scope():

        net.add(mx.gluon.rnn.LSTM(nhidden[0], layout = 'NTC', dropout = dropout))
        
        net.add(mx.gluon.rnn.LSTM(nhidden[1], layout = 'NTC', dropout = dropout))
        
        net.add(mx.gluon.rnn.LSTM(1, layout = 'NTC'))

        #net.add(mx.gluon.nn.Dense(1, activation = 'sigmoid'))
        #if mode == 'Training':

        #    net.add(mx.gluon.nn.Dropout(.5))
        
        #net.add(mx.gluon.nn.Activation('sigmoid'))

    return net

def birnn(nhidden, weight, device, mode, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    if mode != 'train':

        dropout = 0
    
    with net.name_scope():

        #net.add(mx.gluon.rnn.LSTM(nhidden[0], layout = 'NTC', bidirectional = True, dropout = dropout))
        
        net.add(mx.gluon.rnn.LSTM(nhidden[1], layout = 'NTC', bidirectional = True, dropout = dropout))
        
       # net.add(mx.gluon.rnn.(1, layout = 'NTC'))

        net.add(mx.gluon.rnn.LSTM(1, layout = 'NTC'))
        
        #net.add(mx.gluon.nn.Activation('sigmoid'))

    return net
    
    """ 
    with self.name_scope():
        
        self.dense0 = nn.Dense(128)
        
        self.dense1 = nn.Dense(64)
        
        self.dense2 = nn.Dense(10)

    def forward(self, x):
        
        x = nd.relu(self.dense0(x))
        
        x = nd.relu(self.dense1(x))
    
        return self.dense2(x)            #p.data()[:] =  
    """

        
def encoder_decoder(nhidden, weight, device, mode, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    if mode != 'train':

        dropout = 0
    
    with net.name_scope('encoder'):

        #net.add(mx.gluon.rnn.LSTM(nhidden[0], layout = 'NTC', bidirectional = True, dropout = dropout))
        
        net.add(mx.gluon.rnn.LSTM(nhidden[1], layout = 'NTC', bidirectional = True, dropout = dropout))

       # net.add(mx.gluon.rnn.(1, layout = 'NTC'))

        net.add(mx.gluon.rnn.LSTM(1, layout = 'NTC'))
        
    with net.name_scope('decoder'):

        #net.add(mx.gluon.rnn.LSTM(nhidden[0], layout = 'NTC', bidirectional = True, dropout = dropout))
        
        net.add(mx.gluon.rnn.LSTM(nhidden[1], layout = 'NTC', bidirectional = True, dropout = dropout))

       # net.add(mx.gluon.rnn.(1, layout = 'NTC'))

        net.add(mx.gluon.rnn.LSTM(1, layout = 'NTC'))
        #net.add(mx.gluon.nn.Activation('sigmoid'))

    return net

def lstm(inputs, n_hidden, n_sequence, weights, biases):

    lstm_cell = rnn.BasicLSTMCell(n_hidden[0])

    outputs, states = rnn.dynamic_rnn(lstm_cell, n_sequence, dtype = tf.float32)

    # Linear activation, using rnn inner loop last output
    return (tf.matmul(outputs[-1], weights['out']) + biases['out'])

def BiLSTM(inputs, weights, biases, nframes, nhidden):

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(nhidden[0], forget_bias = 1.0)
    
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(nhidden[0], forget_bias = 1.0)

    # Get lstm cell output
    states, output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, nframes, dtype = tf.float32)
    
    #return tf.concat(states, axis = 2)
    return tf.unstack(tf.concat(states, axis = 2), axis = 1)

