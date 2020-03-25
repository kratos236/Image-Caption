# from flair.embeddings import ELMoEmbeddings
# option = '/DATA/118/hzhu/image_caption/models/model_option.json'
# file_name = '/DATA/118/hzhu/image_caption/models/elmo_pt_weights.hdf5'
# embeddings = ELMoEmbeddings(options_file=option,weight_file=file_name)
# print("Load ELMo model")

from bilm.training import train, load_options_latest_checkpoint, load_vocab, LanguageModel
from bilm.data import LMDataset, BidirectionalLMDataset
import json
import os
import tensorflow as tf
""" Load a pretrained ELMo model. """
model_path = '/DATA/118/hzhu/image_caption/checkpoint'
print("Loading the CNN from %s..." % model_path)
options, ckpt_file = load_options_latest_checkpoint(model_path)

model = LanguageModel(options, True)

import pprint
variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
pprint.pprint(variables)

print('ELMo loaded')

# def build_elmo(self):
#     print("Building the ELMo...")
#     config = self.config
#     # Setup the placeholders
#     if self.is_train:
#         contexts = self.conv_feats
#         sentences = tf.placeholder(
#             dtype = tf.int32,
#             shape = [config.batch_size, config.max_caption_length])
#         masks = tf.placeholder(
#             dtype = tf.float32,
#             shape = [config.batch_size, config.max_caption_length])
#     else:
#         contexts = tf.placeholder(
#             dtype = tf.float32,
#             shape = [config.batch_size, self.num_ctx, self.dim_ctx])
#         last_memory = tf.placeholder(
#             dtype = tf.float32,
#             shape = [config.batch_size, config.num_lstm_units])
#         last_output = tf.placeholder(
#             dtype = tf.float32,
#             shape = [config.batch_size, config.num_lstm_units])
#         last_word = tf.placeholder(
#             dtype = tf.int32,
#             shape = [config.batch_size])
#
#         # Prepare to run
#         predictions = []
#         if self.is_train:
#             alphas = []
#             cross_entropies = []
#             predictions_correct = []
#             num_steps = config.max_caption_length
#             last_word = tf.zeros([config.batch_size], tf.int32)
#         else:
#             num_steps = 1
#         last_state = last_memory, last_output
#
#     # Setup the option
#     import json
#     with open('/DATA/118/hzhu/image_caption/checkpoint/model_option.json', 'r') as f:
#         options = json.loads(f.read())
#     self.options = options
#     self.is_training = self.is_train
#     self.bidirectional = options.get('bidirectional', False)
#
#     # use word or char inputs?
#     self.char_inputs = 'char_cnn' in self.options
#
#     # for the loss function
#     self.share_embedding_softmax = options.get(
#         'share_embedding_softmax', False)
#     if self.char_inputs and self.share_embedding_softmax:
#         raise ValueError("Sharing softmax and embedding weights requires "
#                          "word input")
#
#     self.sample_softmax = options.get('sample_softmax', True)
#
#     print("ELMo built.")
#
#     # Compute the loss for this step, if necessary
#     # size of input options
#     n_tokens_vocab = self.options['n_tokens_vocab']
#     batch_size = self.options['batch_size']
#     unroll_steps = self.options['unroll_steps']
#
#     # LSTM options
#     lstm_dim = self.options['lstm']['dim']
#     projection_dim = self.options['lstm']['projection_dim']
#     n_lstm_layers = self.options['lstm'].get('n_layers', 1)
#     dropout = self.options['dropout']
#     keep_prob = 1.0 - dropout
#
#     if self.char_inputs:
#         self._build_word_char_embeddings()
#     else:
#         self._build_word_embeddings()
#
#     # now the LSTMs
#     # these will collect the initial states for the forward
#     #   (and reverse LSTMs if we are doing bidirectional)
#     self.init_lstm_state = []
#     self.final_lstm_state = []
#
#     # get the LSTM inputs
#     if self.bidirectional:
#         lstm_inputs = [contexts, self.embedding, self.embedding_reverse]
#     else:
#         lstm_inputs = [contexts, self.embedding]
#
#     # now compute the LSTM outputs
#     cell_clip = self.options['lstm'].get('cell_clip')
#     proj_clip = self.options['lstm'].get('proj_clip')
#
#     use_skip_connections = self.options['lstm'].get(
#         'use_skip_connections')
#     if use_skip_connections:
#         print("USING SKIP CONNECTIONS")
#
#     lstm_outputs = []
#     for lstm_num, lstm_input in enumerate(lstm_inputs):
#         lstm_cells = []
#         for i in range(n_lstm_layers):
#             if projection_dim < lstm_dim:
#                 # are projecting down output
#                 lstm_cell = tf.nn.rnn_cell.LSTMCell(
#                     lstm_dim, num_proj=projection_dim,
#                     cell_clip=cell_clip, proj_clip=proj_clip)
#             else:
#                 lstm_cell = tf.nn.rnn_cell.LSTMCell(
#                     lstm_dim,
#                     cell_clip=cell_clip, proj_clip=proj_clip)
#
#             if use_skip_connections:
#                 # ResidualWrapper adds inputs to outputs
#                 if i == 0:
#                     # don't add skip connection from token embedding to
#                     # 1st layer output
#                     pass
#                 else:
#                     # add a skip connection
#                     lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
#
#             # add dropout
#             if self.is_training:
#                 lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
#                                                           input_keep_prob=keep_prob)
#
#             lstm_cells.append(lstm_cell)
#
#         if n_lstm_layers > 1:
#             lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
#         else:
#             lstm_cell = lstm_cells[0]
#
#         with tf.control_dependencies([lstm_input]):
#             self.init_lstm_state.append(
#                 lstm_cell.zero_state(batch_size, DTYPE))
#             # NOTE: this variable scope is for backward compatibility
#             # with existing models...
#             if self.bidirectional:
#                 with tf.variable_scope('RNN_%s' % lstm_num):
#                     _lstm_output_unpacked, final_state = tf.nn.static_rnn(
#                         lstm_cell,
#                         tf.unstack(lstm_input, axis=1),
#                         initial_state=self.init_lstm_state[-1])
#             else:
#                 _lstm_output_unpacked, final_state = tf.nn.static_rnn(
#                     lstm_cell,
#                     tf.unstack(lstm_input, axis=1),
#                     initial_state=self.init_lstm_state[-1])
#             self.final_lstm_state.append(final_state)
#
#         # (batch_size * unroll_steps, 512)
#         lstm_output_flat = tf.reshape(
#             tf.stack(_lstm_output_unpacked, axis=1), [-1, projection_dim])
#         if self.is_training:
#             # add dropout to output
#             lstm_output_flat = tf.nn.dropout(lstm_output_flat,
#                                              keep_prob)
#         tf.add_to_collection('lstm_output_embeddings',
#                              _lstm_output_unpacked)
#
#         lstm_outputs.append(lstm_output_flat)
#
#     for idx in range(num_steps):
#         # Attention mechanism
#         alpha = self.attend(contexts, last_output)
#         context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
#                                 axis = 1)
#         if self.is_train:
#             tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
#                                  [1, self.num_ctx])
#             masked_alpha = alpha * tiled_masks
#             alphas.append(tf.reshape(masked_alpha, [-1]))
#
#         # Embed the last word
#
#         word_embed = self.embedding
#        # Apply the LSTM
#
#         current_input = tf.concat([context, word_embed], 1)
#         output, state = self.elmo(current_input, last_state)
#         memory, _ = state
#
#         # Decode the expanded output of LSTM into a word
#
#         expanded_output = tf.concat([output,
#                                      context,
#                                      word_embed],
#                                      axis = 1)
#         logits = self.decode(expanded_output)
#         probs = tf.nn.softmax(logits)
#         prediction = tf.argmax(logits, 1)
#         predictions.append(prediction)
#
#         # Compute the loss for this step, if necessary
#         if self.is_train:
#             cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 labels = sentences[:, idx],
#                 logits = logits)
#             masked_cross_entropy = cross_entropy * masks[:, idx]
#             cross_entropies.append(masked_cross_entropy)
#
#             ground_truth = tf.cast(sentences[:, idx], tf.int64)
#             prediction_correct = tf.where(
#                 tf.equal(prediction, ground_truth),
#                 tf.cast(masks[:, idx], tf.float32),
#                 tf.cast(tf.zeros_like(prediction), tf.float32))
#             predictions_correct.append(prediction_correct)
#
#             last_output = output
#             last_memory = memory
#             last_state = state
#             last_word = sentences[:, idx]
#
#         tf.get_variable_scope().reuse_variables()
#     # Compute the final loss, if necessary
#     if self.is_train:
#         cross_entropies = tf.stack(cross_entropies, axis = 1)
#         cross_entropy_loss = tf.reduce_sum(cross_entropies) \
#                              / tf.reduce_sum(masks)
#
#         alphas = tf.stack(alphas, axis = 1)
#         alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
#         attentions = tf.reduce_sum(alphas, axis = 2)
#         diffs = tf.ones_like(attentions) - attentions
#         attention_loss = config.attention_loss_factor \
#                          * tf.nn.l2_loss(diffs) \
#                          / (config.batch_size * self.num_ctx)
#
#         reg_loss = tf.losses.get_regularization_loss()
#
#         total_loss = cross_entropy_loss + attention_loss + reg_loss
#
#         predictions_correct = tf.stack(predictions_correct, axis = 1)
#         accuracy = tf.reduce_sum(predictions_correct) \
#                    / tf.reduce_sum(masks)
#
#     self.contexts = contexts
#     if self.is_train:
#         self.sentences = sentences
#         self.masks = masks
#         self.total_loss = total_loss
#         self.cross_entropy_loss = cross_entropy_loss
#         self.attention_loss = attention_loss
#         self.reg_loss = reg_loss
#         self.accuracy = accuracy
#         self.attentions = attentions
#     else:
#         self.initial_memory = initial_memory
#         self.initial_output = initial_output
#         self.last_memory = last_memory
#         self.last_output = last_output
#         self.last_word = last_word
#         self.memory = memory
#         self.output = output
#         self.probs = probs


