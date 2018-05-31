from mxnet.gluon import nn, Block, rnn
from mxnet import nd

class Encoder(Block):
    
    """编码器"""

    def __init__(self, n_inputs, n_hidden, n_layers = 1, dropout = 0.5):
        
        super(Encoder, self).__init__()
        
        with self.name_scope():
        
            #self.dropout = nn.Dropout(dropout)
            
            self.rnn_1 = rnn.LSTM(n_hidden, n_layers, dropout = dropout, input_size = n_inputs, bidirectional = True)
            
            #self.rnn_1 = rnn.LSTM(n_hidden / 2 / 2, n_layers, dropout = dropout, input_size = n_hidden, bidirectional = True)

    def forward(self, inputs, states):

        # inputs尺寸: (1, num_steps, 256)，emb尺寸: (num_steps, 1, 256)
        
        hidden_1, state_1 = self.rnn_1(inputs, states)

        #hidden_2, last_state_2 = self.rnn_2(, states)

        return hidden_1, state_1

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)

class Decoder(Block):
    
    def __init__(self, n_inputs, n_encoder_state, n_decoder_state, n_decoder_output, n_sequence_max, n_layers = 1, dropout = 0.5):

        super(Decoder, self).__init__()
        
        self.n_sequence_max = n_sequence_max
        
        self.n_encoder_state = n_encoder_state

        self.n_encoder_state = n_decoder_state

        self.n_layers = n_layers

        self.n_inputs = n_inputs
        
        with self.name_scope():

            self.dropout = nn.Dropout(dropout)

            self.attention = nn.Sequential()

            with self.attention.name_scope():

                # 這裏的 input 是 [encoder output [time, batch, feature] + encoder last state [1, batch, feature] broadcast over time

                # encoder last state broadcast [time, batch, feature]

                # concat 2 matrix get => [time, batch, 2 * feature]

                # so the input [2 * feature]  here is pair of (each step feature, last state feature)

                # 把 nhidden (feature) mapping 至 1，也就是 [time, batch, 1]

                # 可以學習到每個時間的貢獻
                
                # 之後在對 time　做 softmax

                # 但是這樣的動作應該只是一次而已，也就是這個 attention 只是 a0(or a1)，後面的 attention weight　還要繼續計算(透過 rnn)
               
                self.attention.add(nn.Dense(n, activation = 'tanh', flatten = False))
                
                self.attention.add(nn.Dense(1, flatten = False))

            self.rnn_cell = rnn.LSTMCell(n_decoder_state * 2, input_size = n_inputs + n_encoder_state * 2) # 

            self.dense_output = nn.Dense(n_decoder_output)
            
    def forward(self, inputs, state, encoder_all_states):
        
        # 当RNN为多层时，取最靠近输出层的单层隐含状态。

        # state is a list [ ndarray ] => list [ nd[layer, batch, nhidden] ]

        # state[0] => ndarray => nd[layer, batch, nhidden]

        # state[0][-1] => ndarray last layer state => nd[batch, nhidden]

        # expand_dims(0) => nd[1, batch, nhidden]

        # single_layer_state => list(nd[1, batch, nhidden])

        #print('eo 2 : ', encoder_outputs.shape)

        print('state : ', state)
        
        encoder_last_state_h = state[0].reshape([1, -1, self.n_encoder_state * 2]) # x2 for bidirection [-1].expand_dims(0) # time major

        encoder_last_state_c = state[1].reshape([1, -1, self.n_encoder_state * 2])#[-1].expand_dims(0)

        print('encoder state c : ', encoder_last_state_c.shape)

        print('encoder state h : ', encoder_last_state_h.shape)

        encoder_last_state_list = [encoder_last_state_h, encoder_last_state_c]
        
        #encoder_outputs = encoder_outputs.reshape((self.max_seq_len, 1, self.encoder_noutput))

        #single_layer_state尺寸: [(1, 1, decoder_hidden_dim)]
        
        #hidden_broadcast尺寸: (max_seq_len, 1, decoder_hidden_dim)
        
        outputs = []
        
        for t in range(inputs.shape[0]):

            if t == 0:

                decoder_last_state_h = encoder_last_state_h; decoder_last_state_c = encoder_last_state_c

            decoder_last_state_broadcast_h = nd.broadcast_axis(decoder_last_state_h, axis = 0, size = self.n_sequence_max)
        
            decoder_last_state_broadcast_c = nd.broadcast_axis(decoder_last_state_c, axis = 0, size = self.n_sequence_max)

            decoder_last_state_broadcast = nd.concat(decoder_last_state_broadcast_h, decoder_last_state_broadcast_c, dim = 2)

            print('decoder_last state broadcast : ', decoder_last_state_broadcast.shape)

            print('decoder_last_state : ', encoder_last_state_list[0].shape)

            # concat all of the encoder state with decoder last state, so we can preidct attention weights
            
            encoder_all_states_and_decoder_last_state_broadcast = nd.concat(encoder_all_states, decoder_last_state_broadcast, dim = 2)

            # computer attention enengy

            energy = self.attention(encoder_all_states_and_decoder_last_state_broadcast) # time major

            # make the engery in range of [0, 1] with softmax

            attention_weight_batch_major = nd.softmax(energy, axis = 0).swapaxes(0, 1).reshape((-1, 1, self.n_sequence_max)) # 對 time 做 softmax

            print('attention_weight_batch_majorattention : ', attention_weight_batch_major.shape)

            encoder_all_states_batch_major = encoder_all_states.swapaxes(0, 1)

            decoder_context = nd.batch_dot(attention_weight_batch_major, encoder_all_states_batch_major).swapaxes(0, 1) # decoder context -> time major

            print('decoder_context : ', decoder_context.shape)

            current_input = inputs[t].expand_dims(0)

            print('current_input : ', current_input.shape)

            # input_and_context尺寸: (1, 1, encoder_hidden_dim + decoder_hidden_dim)
            
            input_and_context = nd.concat(current_input, decoder_context, dim = 2)
     
            # concat_input尺寸: (1, 1, decoder_hidden_dim)
            
            #concat_input = self.rnn_concat_input(input_and_context)

            #concat_input = self.dropout(concat_input)

            # 当RNN为多层时，用单层隐含状态初始化各个层的隐含状态。



            #decoder_state_c = encoder_last_state_c #nd.broadcast_axis(encoder_last_state_c, axis = 0, size = self.n_layers)
            
            #decoder_state_h = decoder_last_state_h #nd.broadcast_axis(encoder_last_state_h, axis = 0, size = self.n_layers)

            #decoder_state_init_list = [decoder_state_h[0], decoder_state_c[0]]

            print('input_and_context : ', input_and_context.shape)

            print('cstate [0]: ', state[0].shape)
            
            print('hstate [1]: ', state[1].shape)

            print('n_encoder_state + n_input : ', self.n_encoder_state + self.n_inputs)

            print('decoder state c : ', decoder_last_state_c.shape)
            
            print('decoder state h : ', decoder_last_state_h.shape)

            decoder_last_state_list = [decoder_last_state_h[0], decoder_last_state_c[0]]

            decoder_current_state, decoder_last_state = self.rnn_cell(input_and_context[0], decoder_last_state_list)

            outputs.append(self.dense_output(self.dropout(decoder_current_state)))
                
            # output尺寸: (1, output_size)，hidden尺寸: [(1, 1, decoder_hidden_dim)]
        
        return outputs

    def begin_state(self, *args, **kwargs):
        
        return self.rnn.begin_state(*args, **kwargs)

class DecoderInitState(Block):

    """解码器隐含状态的初始化"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        
        super(DecoderInitState, self).__init__()
        
        with self.name_scope():
        
            self.dense = nn.Dense(decoder_hidden_dim, in_units = encoder_hidden_dim, activation = "tanh", flatten = False)

    def forward(self, encoder_state):
        
        return [self.dense(encoder_state)]
