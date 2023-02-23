from .base import *

class attn_deep_s2s_model(base_model):
    def __init__(self,
                 exp_title = '',
                 loss='categorical_crossentropy',
                 metrics=['acc'],
                 recurrent_layer='LSTM',
                 embedding_size=50,
                 num_encoder_tokens=None,
                 num_hidden_cells=None,
                 num_decoder_tokens=None,
                 dropout=0.2,
                 optimizer = 'adam',
                 lr = .001,
                 max_target_seq_len=30,
                 decay_step_size = 20,
                 decay_factor = .98,
                 max_input_seq_len = 327                    
                 ):

        self.max_input_seq_len = max_input_seq_len
        super().__init__(
                            exp_title = 'Attention_' + exp_title,
                            loss = loss,
                            metrics = metrics,
                            recurrent_layer = recurrent_layer,
                            embedding_size = embedding_size,
                            num_encoder_tokens = num_encoder_tokens,
                            num_hidden_cells = num_hidden_cells,
                            num_decoder_tokens = num_decoder_tokens,
                            dropout = dropout,
                            optimizer = optimizer,
                            lr = lr,
                            decay_step_size = decay_step_size,
                            decay_factor = decay_factor,
                            max_target_seq_len = max_target_seq_len,
                    )

    # Function: seq2seq_model
    def build(self):
        self.encoder_inputs = Input(shape=(self.max_input_seq_len,), name='encoder_inputs')
        self.enc_ad_input = Input(shape=(None,), name = 'antigenic_distance input')
        en_x = Embedding(self.num_encoder_tokens, self.embedding_size, name = 'encoder_embedding')(self.encoder_inputs)
        addition = Add( name='Addition_layer')
        en_x = addition([en_x, self.enc_ad_input])
            
        if self.recurrent_layer == 'Attn_Deep_BI_LSTM':
            '''
            #layer_1
            encoder_1 =  Bidirectional(LSTM(self.num_hidden_cells, return_state=True, return_sequences=True, dropout=self.dropout))
            encoder_outputs_1, f_state_h1, f_state_c1, b_state_h1, b_state_c1 = encoder_1(en_x)
            '''
            # layer_2
            encoder_2 =  Bidirectional(LSTM(self.num_hidden_cells, return_state=True, return_sequences=True, dropout=self.dropout))
            encoder_outputs_2, f_state_h2, f_state_c2, b_state_h2, b_state_c2 = encoder_2(en_x)
            # layer_3
            encoder =  Bidirectional(LSTM(self.num_hidden_cells, return_state=True, return_sequences=True, dropout=self.dropout))
            self.encoder_outputs, f_state_h, f_state_c, b_state_h, b_state_c = encoder(encoder_outputs_2)
            # concatenation
            state_h = Concatenate()([f_state_h, b_state_h])
            state_c = Concatenate()([f_state_c, b_state_c])
            self.encoder_states = [state_h, state_c]
        elif self.recurrent_layer == 'Attn_Deep_BI_GRU':
            # GRU layer_1
            encoder_1 = Bidirectional(GRU(self.num_hidden_cells, return_state=True, return_sequences=True, dropout=self.dropout))
            encoder_outputs_1, f_encoder_states_1, b_encoder_states_1  = encoder_1(en_x)
            # GRU layer_2
            encoder = Bidirectional(GRU(self.num_hidden_cells, return_state=True, return_sequences=True, dropout=self.dropout))
            self.encoder_outputs, f_encoder_states, b_encoder_states = encoder(encoder_outputs_1)
            self.encoder_states = Concatenate()([f_encoder_states, b_encoder_states])
         


        # decoder
        self.decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        self.dex = Embedding(self.num_decoder_tokens, self.embedding_size, name='decoder_embedding')
        # dec_ad_input = Input(shape=(None,))
        final_dex = self.dex(self.decoder_inputs)
        # final_dex = final_dex + dec_ad_input
        if self.recurrent_layer == 'Attn_Deep_BI_LSTM':
            self.decoder_lstm = LSTM(self.num_hidden_cells*2, return_sequences=True, return_state=True, dropout=self.dropout)
            self.decoder_outputs, _, _ = self.decoder_lstm(final_dex, initial_state=self.encoder_states) 
        elif self.recurrent_layer == 'Attn_Deep_BI_GRU':
            self.decoder_lstm = GRU(self.num_hidden_cells*2, return_sequences=True, return_state=True, dropout=self.dropout)
            self.decoder_outputs, _ = self.decoder_lstm(final_dex, initial_state=self.encoder_states)
        
        #Attention
        self.attention = AttentionLayer(name='Attention')
        attention_output, _ = self.attention([self.encoder_outputs, self.decoder_outputs])
        
        # concatenation 
        decoder_attention_outputs = Concatenate(axis=-1, name='Concat_layer')([self.decoder_outputs, attention_output])
        
        self.decoder_dense_softmax = TimeDistributed(Dense(self.num_decoder_tokens, activation='softmax', name='Dense'))
        self.decoder_outputs = self.decoder_dense_softmax(decoder_attention_outputs)
        #model = Model([self.encoder_inputs, self.decoder_inputs, self.enc_ad_input], decoder_outputs)
        

        
        if self.optimizer == 'adam':
            opt = Adam(learning_rate=self.lr)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr)

        self.model = Model([self.encoder_inputs, self.decoder_inputs, self.enc_ad_input], self.decoder_outputs)
        self.model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        
        # return self.model, self.encoder_inputs, self.enc_ad_input, self.encoder_states, self.decoder_inputs, \
        #        self.dex, self.decoder_lstm, self.decoder_dense_softmax

    def inf_encoder_model(self):
        '''
        this function will take
        encoder input layer
        encoder antigenic distance input layer
        and encoder states as inputs
        to create an encoder model for inference
        '''
        self.encoder_model = Model([self.encoder_inputs, self.enc_ad_input], [self.encoder_outputs, self.encoder_states])
        # return self.encoder_model

    def inf_decoder_model(self):
                          # , num_hidden_cells,
                          # decoder_lstm,
                          # decoder_dense_softmax

        '''
        this function takes number LSTM of hidden cells
        maximum legnth for target sequence
        embedding size
        decoder input layer
        decoder embedding layer
        decoder lstm layer
        decoder dense layer
        which resulted from  seq2seq_model function
        and returns decoder_model for inference
        '''
        if self.recurrent_layer == 'Attn_Deep_BI_LSTM':
            decoder_state_input_h = Input(shape=(self.num_hidden_cells*2,))
            decoder_state_input_c = Input(shape=(self.num_hidden_cells*2,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            encoder_states_input = Input(shape=(self.max_input_seq_len, self.num_hidden_cells*2))
        elif self.recurrent_layer == 'Attn_Deep_BI_GRU':
            decoder_states_inputs = Input(shape=(self.num_hidden_cells*2,))
            encoder_states_input = Input(shape=(self.max_input_seq_len,self.num_hidden_cells*2))
        # dec_input_ad = Input(shape=(max_target_seq_len, embedding_size))

        final_dex2 = self.dex(self.decoder_inputs)
        # final_dex2 = final_dex2 + dec_input_ad

        if self.recurrent_layer  == 'Attn_Deep_BI_LSTM':
            decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
            decoder_states2 = [state_h2, state_c2]
        elif self.recurrent_layer  == 'Attn_Deep_BI_GRU':
            decoder_outputs2, decoder_states2 = self.decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        #Attention
        attention_outputs2, _ = self.attention([encoder_states_input, decoder_outputs2])
        
        #Concatentaion
        decoder_attention_outputs2 = Concatenate()([decoder_outputs2, attention_outputs2])
        
        decoder_outputs2 = self.decoder_dense_softmax(decoder_attention_outputs2)
        if self.recurrent_layer  == 'Attn_Deep_BI_LSTM':
            self.decoder_model = Model([self.decoder_inputs] + [encoder_states_input, decoder_states_inputs], [decoder_outputs2] + decoder_states2)
        elif self.recurrent_layer  == 'Attn_Deep_BI_GRU':
            self.decoder_model = Model([self.decoder_inputs] + [encoder_states_input, decoder_states_inputs], [decoder_outputs2] + [decoder_states2])

        # return self.decoder_model


    def decode_sequence(self,
                        input_seq, enc_input_ad, max_len_decoded_sentence,
                        target_token_index, reverse_target_index
                        # encoder_model, decoder_model,
                        # RNN = RNN_net
                        ):

        # input_seq, enc_input_ad, max_len_decoded_sentence, encoder_model, decoder_model, target_token_index,
        #                 reverse_target_index, RNN="LSTM"):
        '''
        this function runs the inference procedure it takes
        an input sequence, encoder input antigenic distance and maximum length of decoded sentence
        the return is the infered sequence
        '''
        # Encode the input as state vectors.
        encoder_output, states_value = self.encoder_model.predict([input_seq, enc_input_ad])
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = target_token_index['<BOS>']

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            if self.recurrent_layer == 'Attn_Deep_BI_LSTM':
                output_tokens, h, c = self.decoder_model.predict([target_seq] + [encoder_output, states_value])
            elif self.recurrent_layer == 'Attn_Deep_BI_GRU':
                output_tokens, states = self.decoder_model.predict([target_seq] + [encoder_output, states_value])
                # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_index[sampled_token_index]
            decoded_sentence += ' ' + sampled_char
            # print(decoded_sentence)
            # Exit condition: either hit max length
            # or find stop character.

            if (sampled_char == '<EOS>'
                    or
                    len(decoded_sentence) > max_len_decoded_sentence
            ):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            if self.recurrent_layer == 'Attn_Deep_BI_LSTM':
                states_value = [h, c]
            elif self.recurrent_layer == 'Attn_Deep_BI_GRU':
                states_value = [states]

        return decoded_sentence

    def train(self,encoder_input_data, decoder_input_data, enc_antigenic_distance,decoder_target_data,
              epochs,batch_size,val_split,patience
              ):
        lr_sched = step_decay_schedule(initial_lr=self.lr , decay_factor=self.decay_factor , step_size=self.decay_step_size)
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience,
                                       verbose=1, mode='max',
                                       restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=self.exp_title + '_best_model.h5',
                                     monitor='val_acc',
                                     verbose=2, save_best_only=True, mode='max')

        callbacks_list = [early_stopping, checkpoint, lr_sched]
        self.history = self.model.fit([encoder_input_data, decoder_input_data, enc_antigenic_distance],
                            decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=val_split,
                            callbacks=callbacks_list)
        # dump('history_file.pkl',self.history)