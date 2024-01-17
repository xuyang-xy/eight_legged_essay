#!/usr/bin/env python
# -*- coding: gbk -*-

import os
import numpy as np
import tensorflow as tf

"""
pairwise deepfm
"""
class DeepFM():
    def __init__(self,
                 field_size,
                 feature_size,
                 embedding_size,
                 dnn_dims,
                 dropout_rate,
                 batch_norm=0,
                 l2_reg=0.0,
                 learning_rate=0.001
                 ):

        # set model parameters
        self.field_size = field_size            # feature field cnt: F
        self.feature_size = feature_size        # feature range cnt: M
        self.embedding_size = embedding_size    # feature embedding dim: K
        self.dnn_dims = dnn_dims
        self.dropout_rate = dropout_rate
        # set tarin parameters
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        # set placeholders
        self.feature_index_pos = tf.placeholder(tf.int64, shape=[None, None], name="feature_index_pos") # int64 for rankserver
        self.feature_index_neg = tf.placeholder(tf.int64, shape=[None, None], name="feature_index_neg")
        self.dropout_keep = tf.placeholder_with_default(1.0, shape=[], name="dropout_keep")
        self.label = tf.placeholder(tf.float32, shape=[None,1], name="label")
        # set graph
        self._init_graph()

    def _inference(self, feat_index):
        """
        input:  feat_index
        return: logits inferenced by deepfm
        """
        # embedding layer
        embeddings = tf.nn.embedding_lookup(self.weights["embedding_tensor"], feat_index) # [None, F, K]
        # print "embedding shape", self.embeddings.shape

        # dnn part
        y_deep = tf.reshape(embeddings, shape=[-1, self.field_size * self.embedding_size]) # [None, FK]
        y_deep = tf.nn.dropout(y_deep, self.dropout_keep) # [None, FK]
        for i in range(0, len(self.dnn_dims)):
            y_deep = tf.add(tf.matmul(y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i]) # None * layer[i] * 1
            y_deep = tf.nn.relu(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_keep) # dropout at each Deep layer

        # fm part
        # first order
        y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], feat_index)
        y_first_order = tf.reduce_sum(y_first_order, 2)
        # sum_square part
        summed_features_emb = tf.reduce_sum(embeddings, 1)
        summed_features_emb_square = tf.square(summed_features_emb)

        # square_sum part
        squared_features_emb = tf.square(embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        # second order
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

        # concatenate deep & fm
        concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)    
        logits = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
        return logits


    def _init_graph(self):
        """init_graph"""
        self.weights = self._initialize_weights()
        # inference res
        self.pos_out = self._inference(self.feature_index_pos)
        self.neg_out = self._inference(self.feature_index_neg)
        self.prob = tf.nn.sigmoid(self.pos_out-self.neg_out)
        self.loss = tf.reduce_mean(tf.losses.log_loss(self.label, self.prob))
        
        self.pred = tf.nn.sigmoid(self.pos_out)
        self.pred_out = tf.reshape(self.pred, shape=[-1],name="pred_ctr")
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss, global_step=self.global_step)

        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        
        self.total_parameters = total_parameters
        self.saver = tf.train.Saver(max_to_keep=None)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = tf.layers.batch_normalization(x,  training=True, reuse=None, trainable=True, name=scope_bn)
        bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, trainable=True, name=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _initialize_weights(self):
        weights = dict()
        # fm layer
        weights["embedding_tensor"] = tf.Variable( tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name="feature_embeddings")
        weights["feature_bias"] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")

        # deep layers
        num_layer = len(self.dnn_dims)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.dnn_dims[0]))
        weights["layer_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.dnn_dims[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_dims[0])), dtype=np.float32)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.dnn_dims[i-1] + self.dnn_dims[i]))
            weights["layer_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.dnn_dims[i-1], self.dnn_dims[i])),dtype=np.float32)
            weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_dims[i])),dtype=np.float32)

        # connect layer
        connect_size = self.field_size + self.embedding_size + self.dnn_dims[-1]
        glorot = np.sqrt(2.0 / (connect_size + 1))
        weights["concat_projection"] = tf.Variable(tf.random_normal([connect_size, 1], 0.0, glorot), dtype=tf.float32)
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def train_batch(self, sess, train_iter):
        queryinfo, posdocinfo, negdocinfo, feature_pos, feature_neg, label = sess.run(train_iter)
        feed_dict = { self.feature_index_pos: feature_pos,
                      self.feature_index_neg: feature_neg,
                      self.dropout_keep: 1.0-self.dropout_rate,
                      self.label: label
                    }

        _, trainloss, globalstep, pospred= sess.run([self.optimizer, self.loss, self.global_step, self.pred_out], feed_dict=feed_dict)

        return trainloss, globalstep, pospred

    def test_batch(self, sess, test_data):
        batch_label = test_data[0]
        batch_queryinfo = test_data[1]
        batch_docinfo = test_data[2]
        batch_featrue = test_data[3]

        feed_dict = {self.feature_index_pos: batch_featrue}
        pospred = sess.run(self.pred_out,feed_dict=feed_dict)
        
        return batch_label, batch_queryinfo, batch_docinfo, pospred
    
    def save(self, sess, checkpoint_dir, model_dir, model_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        # self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=None)
        print 'checkpoint_dir: ', os.path.join(checkpoint_dir, model_name)

    def load(self,sess, checkpoint_dir, model_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print(" [*] Reading checkpoints..."), checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print 'ckpt_name: ', ckpt_name, os.path.join(checkpoint_dir, ckpt_name)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__ == '__main__':
    """for test"""
    deepfm = DeepFM(
                 field_size=56,
                 feature_size=1373,
                 embedding_size=8,
                 dnn_dims=[300,300,300],
                 dropout_rate=0.5,
                 batch_norm=0,
                 l2_reg=0.0,
                 learning_rate=0.001
                 )