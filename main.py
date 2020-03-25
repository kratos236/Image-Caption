#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
import os
from tqdm import tqdm
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher
from bilm.training import print_variable_summary,average_gradients,summary_gradient_updates,clip_by_global_norm_summary,clip_grads,_get_feed_dict_from_X
import numpy as np
import time
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_string('continue_file', None,
                       'If sepcified, continue the train process from latest model')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    restart_ckpt_file = FLAGS.continue_file
    if FLAGS.phase == 'train':
        data = prepare_train_data(config)
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            model_option = '/DATA/118/hzhu/image_caption/checkpoint/model_option.json'
            import json
            with open(model_option, 'r') as fin:
                options = json.load(fin)
            # set up the optimizer
            lr = options.get('learning_rate', 0.2)
            opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                            initial_accumulator_value=1.0)

            # calculate the gradients on each GPU
            tower_grads = []
            models = []
            n_gpus = config.n_gpus
            train_perplexity = tf.get_variable(
                'train_perplexity', [],
                initializer=tf.constant_initializer(0.0), trainable=False)
            norm_summaries = []
            for k in range(n_gpus):
                with tf.device('/gpu:%d' % k):
                    with tf.variable_scope('lm', reuse=k > 0):
                        # calculate the loss for one model replica and get
                        #   lstm states
                        model = CaptionGenerator(config)
                        loss = model.total_loss
                        models.append(model)
                        # get gradients
                        grads = opt.compute_gradients(
                            loss * options['unroll_steps'],
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                        )
                        tower_grads.append(grads)
                        # keep track of loss across all GPUs
                        train_perplexity += loss

            print_variable_summary()

            # calculate the mean of each gradient across all GPUs
            grads = average_gradients(tower_grads, options['batch_size'], options)
            grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
            norm_summaries.extend(norm_summary_ops)

            # log the training perplexity
            train_perplexity = tf.exp(train_perplexity / n_gpus)
            perplexity_summmary = tf.summary.scalar(
                'train_perplexity', train_perplexity)

            # some histogram summaries.  all models use the same parameters
            # so only need to summarize one
            histogram_summaries = [
                tf.summary.histogram('token_embedding', models[0].embedding)
            ]
            # tensors of the output from the LSTM layer
            lstm_out = tf.get_collection('lstm_output_embeddings')
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
            if options.get('bidirectional', False):
                # also have the backward embedding
                histogram_summaries.append(
                    tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

            # apply the gradients to create the training operation
            train_op = opt.apply_gradients(grads, global_step=global_step)

            # histograms of variables
            for v in tf.global_variables():
                histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

            # get the gradient updates -- these aren't histograms, but we'll
            # only update them when histograms are computed
            histogram_summaries.extend(
                summary_gradient_updates(grads, opt, lr))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            summary_op = tf.summary.merge(
                [perplexity_summmary] + norm_summaries
            )
            hist_summary_op = tf.summary.merge(histogram_summaries)

            init = tf.initialize_all_variables()

        # do the training loop
        bidirectional = options.get('bidirectional', False)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init)

            # load the checkpoint data if needed
            if restart_ckpt_file is not None:
                loader = tf.train.Saver()
                loader.restore(sess, restart_ckpt_file)
            now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            os.mkdir(config.summary_dir + '/' + now_time)
            summary_writer = tf.summary.FileWriter(config.summary_dir + '/' + now_time, sess.graph)

            # For each batch:
            # Get a batch of data from the generator. The generator will
            # yield batches of size batch_size * n_gpus that are sliced
            # and fed for each required placeholer.
            #
            # We also need to be careful with the LSTM states.  We will
            # collect the final LSTM states after each batch, then feed
            # them back in as the initial state for the next batch

            batch_size = config.batch_size
            unroll_steps = config.max_caption_length
            n_train_tokens = options.get('n_train_tokens', 768648884)
            n_tokens_per_batch = batch_size * unroll_steps * n_gpus
            n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
            n_batches_total = options['n_epochs'] * n_batches_per_epoch
            # print("Training for %s epochs and %s batches" % (
            #     options['n_epochs'], n_batches_total))

            # get the initial lstm states
            init_state_tensors = []
            final_state_tensors = []
            for model in models:
                init_state_tensors.extend(model.init_lstm_state)
                final_state_tensors.extend(model.final_lstm_state)


            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }

            feed_dict.update({
                model.images:
                    np.zeros([config.batch_size] + model.image_shape, dtype=np.float32)
                for model in models
            })

            feed_dict.update({
                model.masks:
                    np.zeros([batch_size, unroll_steps], dtype=np.float64)
                for model in models
            })

            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                })

                feed_dict.update({
                    model.masks_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.float64)
                    for model in models
                })

            init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

            t1 = time.time()
            for i in range(config.num_epochs):
                for j in range(int(data.num_batches/n_gpus)):
                    # slice the input in the batch for the feed_dict
                    feed_dict = {t: v for t, v in zip(
                        init_state_tensors, init_state_values)}
                    for k in range(n_gpus):
                        model = models[k]
                        try:
                            batch = data.next_batch()
                        except:
                            data.reset()
                            batch = data.next_batch()
                        image_files, sentences, masks = batch
                        images = model.image_loader.load_images(image_files)
                        _feed_dict = {}
                        _feed_dict[model.images] = images
                        _feed_dict[model.token_ids] = sentences
                        _feed_dict[model.token_ids_reverse] = [i[::-1] for i in sentences]
                        _feed_dict[model.masks] = masks
                        _feed_dict[model.masks_reverse] = [j[::-1] for j in masks]
                        feed_dict.update(_feed_dict)
                        # feed_dict = {model.images: images,
                        #      model.token_ids: sentences,
                        #      model.token_ids_reverse: [i[::-1] for i in sentences],
                        #      model.masks: masks,
                        #      model.masks_reverse: [j[::-1] for j in masks]}

                    # This runs the train_op, summaries and the "final_state_tensors"
                    #   which just returns the tensors, passing in the initial
                    #   state tensors, token ids and next token ids
                    step = i*data.num_batches+j
                    if step % 1250 != 0:
                        ret = sess.run(
                            [train_op, summary_op, train_perplexity, model.summary, model.cross_entropy_loss, model.total_loss, model.accuracy] +
                            final_state_tensors,
                            feed_dict=feed_dict
                        )

                        # first three entries of ret are:
                        #  train_op, summary_op, train_perplexity
                        # last entries are the final states -- set them to
                        # init_state_values
                        # for next batch
                        init_state_values = ret[7:]
                        summary_writer.add_summary(ret[3], step)
                        print('Epoch (%s / %s ) Batch ( %s / %s) CE loss: %s Total loss: %s Accuracy:%s' % (i,config.num_epochs,j,int(data.num_batches/n_gpus), ret[4], ret[5], ret[6]))
                    else:
                        # also run the histogram summaries
                        ret = sess.run(
                            [train_op, summary_op, train_perplexity, hist_summary_op] +
                            final_state_tensors,
                            feed_dict=feed_dict
                        )
                        init_state_values = ret[4:]

                    if (step % 3 * int(data.num_batches / n_gpus) == 0):
                        # save the model
                        checkpoint_path = os.path.join(config.save_dir+now_time, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step)

                    if step % 1250 == 0:
                        summary_writer.add_summary(ret[3], step)
                    if step % 100 == 0:
                        # write the summaries to tensorboard and display perplexity
                        summary_writer.add_summary(ret[1], step)
                        print("Batch %s, train_perplexity=%s" % (step, ret[2]))
                        print("Total time: %s" % (time.time() - t1))

                    if step == n_batches_total:
                        # done training!
                        break

                data.reset()
    else :
        with tf.Session(config=gpu_config) as sess:
            if FLAGS.phase == 'eval':
                # evaluation phase
                coco, data, vocabulary = prepare_eval_data(config)
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.get_default_graph().finalize()
                model.eval(sess, coco, data, vocabulary)

            else:
                # testing phase
                data, vocabulary = prepare_test_data(config)
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.get_default_graph().finalize()
                model.test(sess, data, vocabulary)
if __name__ == '__main__':
    tf.app.run()
