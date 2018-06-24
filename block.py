from mxnet.gluon import nn, Block, rnn
from mxnet import nd
import numpy as np

class Encoder(Block):

    def __init__(self, n_inputs, n_hidden, n_layers = 1, dropout = 0.5):
        
        super(Encoder, self).__init__()
        
        with self.name_scope():
        
            self.encoder = rnn.LSTM(n_hidden, n_layers, dropout = dropout, input_size = n_inputs, bidirectional = True)

    def forward(self, inputs, states):

        all_states, last_state = self.encoder(inputs, states)

        return last_state, all_states

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)

class Decoder(Block):
    
    def __init__(self, n_inputs, n_encoder_state, n_decoder_state, n_decoder_output, n_sequence_max, n_alignment, n_layers = 1, dropout = 0.5):

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

                self.attention.add(nn.Dense(n_alignment, activation = 'relu', flatten = False))
                
                self.attention.add(nn.Dense(1, activation = 'sigmoid', flatten = False))

            self.decoder = rnn.LSTMCell(n_decoder_state * 2, input_size = n_inputs + n_encoder_state * 2) # 

            self.dense_output = nn.Dense(n_decoder_output, flatten = False)
            
    def forward(self, inputs, state, encoder_all_states, n_sequence_max_batch, mode = 'Train'):
        
        encoder_last_state_h = state[0].reshape([1, -1, self.n_encoder_state * 2]) # x2 for bidirection [-1].expand_dims(0) # time major

        encoder_last_state_c = state[1].reshape([1, -1, self.n_encoder_state * 2])#[-1].expand_dims(0)

        encoder_last_state_list = [encoder_last_state_h, encoder_last_state_c]
        
        #outputs = nd.zeros(inputs.shape[:-1] + (1,), inputs.context)
        
        #outputs = nd.zeros([n_sequence_max_batch, inputs.shape[1], 1], inputs.context)

        outputs = []

        current_input = None
        
        for t in range(n_sequence_max_batch): 

        #for t in range(inputs.shape[0]): 
            
            # input is already padding according to n_squence_max of all data, but we only have to forward the n_sequencet_max of current batch times
            
            if t == 0:

                decoder_last_state_h = encoder_last_state_h; decoder_last_state_c = encoder_last_state_c

            decoder_last_state_broadcast_h = nd.broadcast_axis(decoder_last_state_h, axis = 0, size = self.n_sequence_max)
        
            decoder_last_state_broadcast_c = nd.broadcast_axis(decoder_last_state_c, axis = 0, size = self.n_sequence_max)

            decoder_last_state_broadcast = nd.concat(decoder_last_state_broadcast_h, decoder_last_state_broadcast_c, dim = 2)

            # concat all of the encoder state with decoder last state, so we can preidct attention weights
            
            encoder_all_states_and_decoder_last_state_broadcast = nd.concat(encoder_all_states, decoder_last_state_broadcast, dim = 2)

            # computer attention enengy

            energy = self.attention(encoder_all_states_and_decoder_last_state_broadcast) # time major

            # make the engery in range of [0, 1] with softmax

            attention_weight_batch_major = nd.softmax(energy, axis = 0).swapaxes(0, 1).reshape((-1, 1, self.n_sequence_max)) # 對 time 做 softmax

            encoder_all_states_batch_major = encoder_all_states.swapaxes(0, 1)

            decoder_context = nd.batch_dot(attention_weight_batch_major, encoder_all_states_batch_major).swapaxes(0, 1) # decoder context -> time major

            # get current input from all of the inputs and expand the time dimension

            if mode == 'Train':

                current_input = inputs[t].expand_dims(0) 

            elif mode == 'Inference':

                current_input = outputs[-1] # using last decoder output for inference mode

            input_and_context = nd.concat(current_input, decoder_context, dim = 2)

            decoder_last_state_list = [decoder_last_state_h[0], decoder_last_state_c[0]]

            decoder_current_state, decoder_last_state = self.decoder(input_and_context[0], decoder_last_state_list)

            outputs.append(self.dense_output(self.dropout(decoder_current_state)))

        return outputs#encoder_all_states_and_decoder_last_state_broadcast#attention_weight_batch_major#outputs#decoder_current_state

    def begin_state(self, *args, **kwargs):
        
        return self.rnn.begin_state(*args, **kwargs)

class EDWA(Block):

    def __init__(self, n_input, n_encoder_hidden, n_decoder_hidden, n_output, n_sequence_max, n_alignment, dropout = 0.5):
        
        super(EDWA, self).__init__()

        self.n_sequence_max = n_sequence_max
        
        with self.name_scope():
        
            self.e = (Encoder(n_input, n_encoder_hidden, 1, 0.5))

            self.d = (Decoder(n_input, n_encoder_hidden, n_decoder_hidden, n_output, n_sequence_max, n_alignment, dropout = dropout))

    def forward(self, inputs, states):
            
        t, n, c = inputs.shape

        pad = nd.zeros([self.n_sequence_max - t, n, c], ctx = inputs.context)

        inputs = nd.concat(inputs, pad, dim = 0)

        last_state, all_states = self.e(inputs, states)

        outputs = self.d(inputs, last_state, all_states, t)

        nd_outputs = outputs[0]
        
        for i in range(1, len(outputs)):

            nd_outputs = nd.concat(nd_outputs, outputs[i], dim = 0)

        return all_states

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

class Wave(Block):

    def __init__(self,
            n_input_states,
            n_output_states,
            size,
            device,
            last = False,
            wave = None,
            nearest = False,
            config = None,
            overlap = True,
            layer = ''):

        """
        
        overlap 
            
            False : 
            
                if kernel is overlapped

                    overlap

        """
        
        super(Wave, self).__init__()

        self.device = device

        self.layer = layer

        self.size = size

        print('kernel : ', self.size['kernel'])

        self.size['context'] = int((self.size['kernel'] - 1) / 2)
        
        self.overlap = overlap

        self.last = last

        self.fc = nn.Sequential()

        if last:

            self.wave = wave

        with self.fc.name_scope():

            if last:

                self.fc.add(nn.Dense(n_output_states, activation = 'sigmoid', flatten = True))

            else:

                self.fc.add(nn.Dense(n_output_states, activation = 'relu', flatten = True))

                self.fc.add(nn.BatchNorm(axis = 1, center = True, scale = True))


    @property
    
    def patch(self):

        return self.size['dilate'] * self.size['context'] * 2 # contains both direction

    def nearest_context(self, currnet_input, states, concat_list, nearest_k, current_time, total_time):

        total_time = [total_time]

        for state, wave in zip(states[1:], self.wave):

            nearest_next = current_time / wave.authorised_stride + (0 if total_time[-1] % authorised_stride > wave.context_size * wave.dilate_size else 1)

            nearest_prev = nearest_next - 1

            next_idx = (concat_list + 1)

            prev_idx = (concat_list - 1)

            concat_list.insert(prev_idx, nearest_prev)

            concat_list.insert(next_idx, nearest_next)

        return concat_list
            
    def context(self, inputs, decoder_inputs = None):

        dilated_context_size = self.size['context'] * self.size['dilate']

        element = (np.arange(- dilated_context_size, dilated_context_size + 1, self.size['dilate']) + self.current_time).astype(int)

        element_insufficient = np.take(element, np.where(element < 0)[0])

        element_exceed = np.take(element, np.where(element >= self.total_time)[0])

        element_authorised = np.take(element, np.where(np.logical_and(element < self.total_time, element >= 0))[0])

        concat_list_exceed = [] if len(element_exceed) == 0 \
                else [nd.zeros([self.n_batch, self.n_states * len(element_exceed)], self.device)]

        concat_list_authorised = [inputs[c] for c in element_authorised]

        concat_list_insufficient = [] if len(element_insufficient) == 0 \
            else [nd.zeros([self.n_batch, self.n_states * len(element_insufficient)], self.device)]

        return concat_list_insufficient + concat_list_authorised + concat_list_exceed

    def context_out(self, inputs, decoder_inputs = None):

        decoder_total_times, decoder_batch, decoder_n_states = decoder_inputs.shape

        inputs_total_times, inputs_batch, inputs_n_states = inputs.shape

        decoder_portion_idx = self.current_time / decoder_total_times

        inputs_center_idx = np.ceil(inputs_total_times * decoder_portion_idx).astype(int)

        dilated_context_size = self.size['context'] * self.size['dilate']

        element = (np.arange(- dilated_context_size, dilated_context_size + 1, self.size['dilate']) + inputs_center_idx).astype(int)

        element_insufficient = np.take(element, np.where(element < 0)[0])

        element_exceed = np.take(element, np.where(element >= inputs_total_times)[0])

        element_authorised = np.take(element, np.where(np.logical_and(element < inputs_total_times, element >= 0))[0])

        concat_list_exceed = [] if len(element_exceed) == 0 \
                else [nd.zeros([self.n_batch, inputs_n_states * len(element_exceed)], self.device)]

        concat_list_authorised = [inputs[c] for c in element_authorised]

        concat_list_insufficient = [] if len(element_insufficient) == 0 \
                else [nd.zeros([self.n_batch, inputs_n_states * len(element_insufficient)], self.device)]

        concat_list_authorised.insert(int(len(concat_list_authorised) / 2), decoder_inputs[self.current_time])

        return concat_list_insufficient + concat_list_authorised + concat_list_exceed

    @property

    def check_overlap(self):

        return True if self.size['stride'] <= 2 * self.size['context'] * self.size['dilate'] else False
    
    def forward(self, inputs, decoder_inputs = None):

        if decoder_inputs is None:
        
            self.total_time, self.n_batch, self.n_states = inputs.shape # if TNC

        elif self.last:

            self.total_time, self.n_batch, self.n_states = decoder_inputs.shape # if TNC

        else:

            ValueError('[!] Not Last Layer ...')

        self.current_time = 0

        if (not self.overlap) and self.check_overlap:

            self.size['authorised_stride'] = 2 * self.size['context'] * self.size['dilate'] + 1

        else:

            self.size['authorised_stride'] = self.size['stride']

        if self.last:

            assert(self.size['stride'] == 1)

            self.size['authorised_stride'] = self.size['stride']

        outputs = []

        start = 0; end = 0;

        #print('self.layer : ', self.layer)

        #print('inputs : ', inputs.shape)

        #print('[*] Authorized Stride : ', self.size['authorised_stride'])

        while self.current_time < self.total_time:

            if decoder_inputs is None:

                concat_list = self.context(inputs)

            elif self.last:

                concat_list = self.context_out(inputs, decoder_inputs)

            concat_data = nd.concat(*concat_list)

            output = self.fc(concat_data).expand_dims(0)
            
            #output = (output + inputs[self.current_time]).expand_dims(0)

            outputs.append(output)

            self.current_time += self.size['authorised_stride']

        outputs = nd.concat(*outputs, dim = 0)

        return outputs

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)


class WaveArch(object):

    inputs = None

    outputs = []

    size = []

    overlap = []

    last = []

class WaveDecoder(Block):

    def __init__(self, arch, n_layers, device = None):

        super(WaveDecoder, self).__init__()
        
        self.n_layers = n_layers

        self.decoder = nn.Sequential()

        with self.decoder.name_scope():

            for l in range(self.n_layers - 1):
                
                self.decoder.add(Wave(arch.inputs[l], arch.outputs[l], arch.size[l],
                    overlap = arch.overlap[l], last = False, device = device, layer = 'wave_{}'.format(l + 1)))

            self.out = Wave(arch.inputs[-1], arch.outputs[-1], arch.size[-1],
                    overlap = arch.overlap[-1], last = True, device = device, layer = 'wave_{}'.format(self.n_layers))
    
    @property

    def patch(self):

        #print('out : ', self.out.patch)

        #print('prev : ', np.prod([d.patch for d in self.decoder]))

        return np.prod([d.patch for d in self.decoder]) * self.out.patch

    def forward(self, decoder_inputs):

        outputs = [decoder_inputs]

        for d in self.decoder:

            outputs.append(d(outputs[-1]))

        outputs.append(self.out(outputs[-1], decoder_inputs))

        return outputs[-1]
    
    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)

# Encoder - WaveDecoder

class EWD(Block):

    def __init__(self, n_input, n_encoder_state, n_wave_output, decoder_arch, n_layers, dropout = 0.5, device = None):
    
        super(EWD, self).__init__()

        with self.name_scope():
        
            self.e = Encoder(n_input, n_encoder_state, 1, dropout)

            self.wd = WaveDecoder(decoder_arch, n_layers, device = device)

    @property
    
    def patch(self):

        return self.wd.patch

    def forward(self, inputs, states):

        encoder_last_states, encoder_all_states = self.e(inputs, states)

        outputs = self.wd(encoder_all_states)

        return outputs

