import tensorflow as tf


class BiLSTMAttention(object):
    """"""

    def __init__(self, config, conv_feats, num_ctx):
        # 定义模型的输入
        contexts = conv_feats
        self.config = config
        sentences = tf.placeholder(
            dtype = tf.int32,
            shape = [config.batch_size, config.max_caption_length])# [32,20,256]
        embedding = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size, config.max_caption_length, config.embeddingSize])
        masks = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size, config.max_caption_length])

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)
        labels = []
        maskss = []
        with tf.name_scope("embedding"):
            embeddingW = tf.get_variable(
                "embeddingW",
                shape=[config.embeddingSize, config.embeddingSize],
                initializer=tf.contrib.layers.xavier_initializer())

            reshapeInputX = tf.reshape(embedding, shape=[-1, config.embeddingSize]) # [640,256]
            self.embeddedWords = tf.matmul(reshapeInputX, embeddingW) #得到了ground truth word的embedding 无需再去查表

            self.embeddedWords = tf.reshape(self.embeddedWords,[config.max_caption_length,config.batch_size,config.embeddingSize])# [20,32,256]
            contexts = tf.layers.dense(inputs=contexts, units=config.embeddingSize, activation=tf.nn.leaky_relu) # 将feature转换到与embeddings相同的size下
            contexts = tf.reshape(contexts,[num_ctx,config.batch_size,config.embeddingSize])# [49,32,256]
            self.embeddedWords = tf.concat([contexts,self.embeddedWords],axis=0)# [69,32,256] 把两个embedding concat起来
            self.embeddedWords = tf.reshape(self.embeddedWords ,shape=[config.batch_size, config.max_caption_length+num_ctx, config.embeddingSize])
            self.embeddedWords = tf.nn.dropout(self.embeddedWords, self.dropoutKeepProb) # [32,69,256]
            # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            # output = self._attention(H)
            outputSize = config.hiddenSizes[-1]
        output = tf.layers.dense(inputs=tf.reshape(H,[config.batch_size,config.hiddenSizes[-1],config.max_caption_length+num_ctx]), units=config.max_caption_length, activation=tf.nn.leaky_relu)#转成句子的长度 相当于一个decoder
        output = tf.reshape(output,[config.batch_size*config.max_caption_length,config.hiddenSizes[-1]]) # [640,256]
        cross_entropies = []
        predictions_correct = []
        predictions = []
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.vocabulary_size],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.vocabulary_size]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            output = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions") # 最后的输出层
            logits = tf.reshape(output,[config.batch_size * config.max_caption_length, config.vocabulary_size])  # [32*20,5002]
            probs = tf.nn.softmax(logits)
            prediction = tf.argmax(logits, 1)



        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            for sentence, mask, pred in zip(sentences, maskss, predictions):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sentence,logits=logits)
                masked_cross_entropy = cross_entropy * mask
                cross_entropies.append(masked_cross_entropy)

                ground_truth = sentence
                prediction_correct = tf.where(
                    tf.equal(pred, ground_truth),
                    tf.cast(mask, tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                tf.get_variable_scope().reuse_variables()

            # Compute the final loss, if necessary
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)
            self.reg_loss = config.reg * l2Loss
            self.loss = cross_entropy_loss + self.reg_loss
            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output