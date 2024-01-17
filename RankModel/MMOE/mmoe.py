#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

"""
mmoe for mul_target_learning in tf.1.x
https://dl.acm.org/doi/10.1145/3219819.3220007
"""
class MMOE():
    def __init__(self,
                 expert_num,
                 task_num,
                 field_size,
                 feature_size,
                 embedding_size,
                 dnn_dims,
                 dropout_rate,
                 l2_reg=0.0,
                 batch_norm=0.0,
                 learning_rate=0.001
                 ):

        # set model parameters
        self.task_num = task_num                # task   num:             T
        self.expert_num = expert_num            # expert num:             E
        self.field_size = field_size            # feature field cnt:      F
        self.feature_size = feature_size        # feature range cnt:      M
        self.embedding_size = embedding_size    # feature embedding dim:  K
        self.dnn_dims = dnn_dims                # expert  encoding  dim:  D
        # set tarin parameters
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        # set placeholders
        self.feature_index_pos = tf.placeholder(tf.int64, shape=[None, None], name="feature_index_pos")
        self.dropout_keep = tf.placeholder_with_default(1.0, shape=[], name="dropout_keep")
        self.label0 = tf.placeholder(tf.float32, shape=[None], name="label0")
        self.label1 = tf.placeholder(tf.float32, shape=[None], name="label1")
        # set graph
        self._init_graph()

    def expert_layer(self, layer_input, expert_index):
        """
        input:  feat_index_embedding
        return: logits inferenced by expert
        """
        y_deep = tf.reshape(layer_input, shape=[-1, self.field_size * self.embedding_size]) # [None, FK]
        y_deep = tf.nn.dropout(y_deep, self.dropout_keep) # [None, FK]
        for i in range(0, len(self.dnn_dims)):
            y_deep = tf.add(tf.matmul(y_deep, self.weights["layer_%d_%d"%(expert_index,i)]), self.weights["bias_%d_%d"%(expert_index,i)]) # None * layer[i] * 1
            y_deep = tf.nn.relu(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_keep) # dropout at each Deep layer
        expert_logit = tf.expand_dims(y_deep, -1)
        return expert_logit

    def gate_layer(self, layer_input, task_index):
        """
        input:  feat_index_embedding
        return: logits inferenced by gate
        """
        input_size = self.field_size * self.embedding_size
        gate_input = tf.reshape(layer_input, shape=[-1, input_size]) # [None, FK]
        gate_x = tf.add(tf.matmul(gate_input, self.weights["gate_w_%d" % task_index]),
                          self.weights["gate_b_%d" % task_index]) # None * layer[i] * 1
        gate_soft = tf.nn.softmax(gate_x)
        expend_gate_output = tf.expand_dims(gate_soft, 1)
        return expend_gate_output

    def clk_tower(self, layer_input):
        """
        input:  weighted_expert
        return: logits inferenced by clktower
        """
        clk_logit = tf.nn.sigmoid(tf.add(tf.matmul(layer_input, self.weights["task_w_0"]), self.weights["task_b_0"]))
        return clk_logit
    
    def time_tower(self, layer_input):
        """
        input:  weighted_expert
        return: logits inferenced by timetower
        """
        time_logit = tf.nn.sigmoid(tf.add(tf.matmul(layer_input, self.weights["task_w_1"]), self.weights["task_b_1"]))
        return time_logit

    def _init_graph(self):
        """init_graph"""
        
        self.weights = self._initialize_weights()
        embeddings = tf.nn.embedding_lookup(self.weights["embedding_tensor"],
                                            self.feature_index_pos) # [None, F, K]
        print "embeddings", embeddings.shape
        self.expert0_out = self.expert_layer(embeddings, 0)
        self.expert1_out = self.expert_layer(embeddings, 1)
        self.expert2_out = self.expert_layer(embeddings, 2)
        self.expert_stack = tf.concat([self.expert0_out, self.expert1_out, self.expert2_out], -1)
        print "expert0_out", self.expert0_out.shape
        print "expert1_out", self.expert1_out.shape
        print "expert2_out", self.expert2_out.shape
        print "expert_stack", self.expert_stack.shape
        self.gate0_weight = self.gate_layer(embeddings, 0)
        self.gate1_weight = self.gate_layer(embeddings, 1)
        print "gate0_weight", self.gate0_weight.shape
        print "gate1_weight", self.gate1_weight.shape
        self.task0_input = tf.reduce_sum(tf.multiply(self.gate0_weight, self.expert_stack), -1)
        self.task1_input = tf.reduce_sum(tf.multiply(self.gate1_weight, self.expert_stack), -1)
        print "task0_input", self.task0_input.shape
        print "task1_input", self.task1_input.shape
        self.task0_out = tf.nn.sigmoid(self.clk_tower(self.task0_input))
        self.task1_out = tf.nn.sigmoid(self.time_tower(self.task1_input))
        
        self.pred_ctr_out = tf.reshape(self.task0_out, shape=[-1],name="pred_ctr")
        self.pred_time_out = tf.reshape(self.task1_out, shape=[-1],name="pred_time")
        print "ctr", self.pred_ctr_out.shape
        print "time", self.pred_time_out.shape
        nlabel0 = tf.reshape(self.label0, shape=[-1,1])
        nlabel1 = tf.reshape(self.label1, shape=[-1,1])
        self.loss0 = tf.reduce_mean(tf.losses.log_loss(nlabel0, self.task0_out))
        self.loss1 = tf.reduce_mean(tf.losses.log_loss(nlabel1, self.task1_out))
        self.loss = self.loss0 + 0.2*self.loss1
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
        # model parameters
        num_layer = len(self.dnn_dims)
        fea_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (fea_size + self.dnn_dims[0]))
        
        weights["embedding_tensor"] = tf.Variable(tf.random_normal([self.feature_size,
                                                                    self.embedding_size],
                                                                    0.0, 0.01),
                                                                   name="feature_embeddings")
        # gate & tower layer
        for t in range(0, self.task_num):
            weights["gate_w_%d"%t] = tf.Variable(np.random.normal(loc=0,
                                                                  scale=glorot,
                                                                  size=(fea_size, self.expert_num)),
                                                                  dtype=np.float32)
            weights["gate_b_%d"%t] = tf.Variable(np.random.normal(loc=0,
                                                                  scale=glorot,
                                                                  size=(1, self.expert_num)),
                                                                  dtype=np.float32)
            weights["tower_w_%d"%t] = tf.Variable(np.random.normal(loc=0,
                                                                  scale=glorot,
                                                                  size=(fea_size, self.expert_num)),
                                                                  dtype=np.float32)
            weights["tower_b_%d"%t] = tf.Variable(tf.constant(0.01), dtype=np.float32)

            glorot = np.sqrt(2.0 / (self.dnn_dims[-1] + 1))
            weights["task_w_%d"%t] = tf.Variable(tf.random_normal([self.dnn_dims[-1], 1], 0.0, glorot), dtype=tf.float32)
            weights["task_b_%d"%t] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # dnn part
        for e in range(0, self.expert_num):
            weights["layer_%d_0"%e] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(fea_size, self.dnn_dims[0])), dtype=np.float32)
            weights["bias_%d_0"%e] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_dims[0])), dtype=np.float32)
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.dnn_dims[i-1] + self.dnn_dims[i]))
                weights["layer_%d_%d"%(e,i)] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.dnn_dims[i-1], self.dnn_dims[i])),dtype=np.float32)
                weights["bias_%d_%d"%(e,i)] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_dims[i])),dtype=np.float32)

        return weights

    def train_batch(self, sess, train_iter):
        queryinfo, posdocinfo, negdocinfo, feature_pos, feature_neg, label = sess.run(train_iter)
        feed_dict = { self.feature_index_pos: feature_pos,
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

if __name__ == "__main__":
    """for test"""
    mmoe = MMOE(expert_num=3,
                task_num=2,
                field_size=2,
                feature_size=4,
                embedding_size=2,
                dnn_dims=[8,8,8],
                dropout_rate=0.5,
                l2_reg=0.0,
                batch_norm=0.0,
                learning_rate=0.001
                )
