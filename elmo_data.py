# import os
# import csv
# import time
# import datetime
# import random
#
# from collections import Counter
# from math import sqrt
#
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import tqdm
# from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher
# from utils.coco.coco import COCO
# from utils.vocabulary import Vocabulary
# from nltk.tokenize import word_tokenize
# from config import Config
#
# # 实例化配置参数对象
# config = Config()
#
# class DataSet(object):
#     def __init__(self,
#                  image_ids,
#                  image_files,
#                  batch_size,
#                  word_idxs=None,
#                  masks=None,
#                  is_train=False,
#                  shuffle=False):
#         self.image_ids = np.array(image_ids)
#         self.image_files = np.array(image_files)
#         self.word_idxs = np.array(word_idxs)
#         self.masks = np.array(masks)
#         self.batch_size = batch_size
#         self.is_train = is_train
#         self.shuffle = shuffle
#         self.setup()
#
#     def _genElmoEmbedding(self):
#         """
#         调用ELMO源码中的dump_token_embeddings方法，基于字符的表示生成词的向量表示。并保存成hdf5文件，文件中的"embedding"键对应的value就是
#         词汇表文件中各词汇的向量表示，这些词汇的向量表示之后会作为BiLM的初始化输入。
#         """
#         dump_token_embeddings(
#             config.vocabFile, config.optionFile, config.weightFile, config.tokenEmbeddingFile)
#
#     def setup(self):
#         """ Setup the dataset. """
#         self.count = len(self.image_ids)
#         self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
#         self.fake_count = self.num_batches * self.batch_size - self.count
#         self.idxs = list(range(self.count))
#         self.reset()
#
#
#     def reset(self):
#         """ Reset the dataset. """
#         self.current_idx = 0
#         if self.shuffle:
#             np.random.shuffle(self.idxs)
#
#
#     def next_batch(self):
#         """ Fetch the next batch. """
#         assert self.has_next_batch()
#
#         if self.has_full_next_batch():
#             start, end = self.current_idx, \
#                          self.current_idx + self.batch_size
#             current_idxs = self.idxs[start:end]
#         else:
#             start, end = self.current_idx, self.count
#             current_idxs = self.idxs[start:end] + \
#                            list(np.random.choice(self.count, self.fake_count))
#
#         image_files = self.image_files[current_idxs]
#         if self.is_train:
#             word_idxs = self.word_idxs[current_idxs]
#             masks = self.masks[current_idxs]
#             self.current_idx += self.batch_size
#             return image_files, word_idxs, masks
#         else:
#             self.current_idx += self.batch_size
#             return image_files
#
#
#     def has_next_batch(self):
#         """ Determine whether there is a batch left. """
#         return self.current_idx < self.count
#
#
#     def has_full_next_batch(self):
#         """ Determine whether there is a full batch left. """
#         return self.current_idx + self.batch_size <= self.count
#
# def process_sentence(self, sentence):
#     """ Tokenize a sentence, and translate each token into its index
#         in the vocabulary. """
#     words = word_tokenize(sentence.lower())
#     word_idxs = [self.word2idx[w] for w in words]
#     return word_idxs
#
# def prepare_elmo_data(config):
#     _dataSource = config.dataSource
#     _stopWordSource = config.stopWordSource
#     _sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
#     _embeddingSize = config.model.embeddingSize
#     _batchSize = config.batchSize
#
#     words = []
#     word2idx = {}
#     word_frequencies = []
#     size = config.vocabulary_size
#     word_counts = {}
#     coco = COCO(config.train_caption_file)
#     coco.filter_by_cap_len(20)
#     sentences = coco.all_captions()
#     for sentence in tqdm(sentences):
#         for w in word_tokenize(sentence.lower()):
#             word_counts[w] = word_counts.get(w, 0) + 1.0
#
#     assert size - 1 <= len(word_counts.keys())
#     words.append('<S>')
#     word2idx['<S>'] = 0
#     word_frequencies.append(1.0)
#     words.append('</S>')
#     word2idx['</S>'] = 1
#     word_frequencies.append(1.0)
#     words.append('<UNK>')
#     word2idx['<UNK>'] = 2
#     word_frequencies.append(1.0)
#
#     word_counts = sorted(list(word_counts.items()),
#                          key=lambda x: x[1],
#                          reverse=True)
#
#     for idx in range(size - 1):
#         word, frequency = word_counts[idx]
#         words.append(word)
#         word2idx[word] = idx + 1
#         word_frequencies.append(frequency)
#
#     word_frequencies = np.array(word_frequencies)
#     word_frequencies /= np.sum(word_frequencies)
#     word_frequencies = np.log(word_frequencies)
#     word_frequencies -= np.max(word_frequencies)
#
#     print("Building the vocabulary...")
#     data = pd.DataFrame({'word': words,
#                          'index': list(range(size)),
#                          'frequency': word_frequencies})
#     data.to_csv('./elmo_vocabulary.csv')
#     print("Vocabulary built.")
#     print("Number of words = %d" % (size))
#
#     coco.filter_by_words(set(words))
#
#     print("Processing the captions...")
#     if not os.path.exists(config.temp_annotation_file):
#         captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
#         image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
#         image_files = [os.path.join(config.train_image_dir,
#                                     coco.imgs[image_id]['file_name'])
#                        for image_id in image_ids]
#         annotations = pd.DataFrame({'image_id': image_ids,
#                                     'image_file': image_files,
#                                     'caption': captions})
#         annotations.to_csv(config.temp_annotation_file)
#     else:
#         annotations = pd.read_csv(config.temp_annotation_file)
#         captions = annotations['caption'].values
#         image_ids = annotations['image_id'].values
#         image_files = annotations['image_file'].values
#
#     if not os.path.exists(config.temp_data_file):
#         word_idxs = []
#         masks = []
#         for caption in tqdm(captions):
#             current_word_idxs_ = process_sentence(caption)
#             current_num_words = len(current_word_idxs_)
#             current_word_idxs = np.zeros(config.max_caption_length,
#                                          dtype=np.int32)
#             current_masks = np.zeros(config.max_caption_length)
#             current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
#             current_masks[:current_num_words] = 1.0
#             word_idxs.append(current_word_idxs)
#             masks.append(current_masks)
#         word_idxs = np.array(word_idxs)
#         masks = np.array(masks)
#         data = {'word_idxs': word_idxs, 'masks': masks}
#         np.save(config.temp_data_file, data)
#     else:
#         data = np.load(config.temp_data_file).item()
#         word_idxs = data['word_idxs']
#         masks = data['masks']
#     print("Captions processed.")
#     print("Number of captions = %d" % (len(captions)))
#     print("Building the dataset...")
#     dataset = DataSet(image_ids,
#                       image_files,
#                       config.batch_size,
#                       word_idxs,
#                       masks,
#                       True,
#                       True)
#     print("Dataset built.")
#     return dataset
