import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class SentenceTokenizer:
    def __init__(self, batch_size, seq_size, dic_size, hidden_size, embedding_size, learning_rate):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dic_size = dic_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate 
    
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.W = tf.get_variable('W', shape=[hidden_size, dic_size], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b', shape=[dic_size], initializer=tf.contrib.layers.xavier_initializer())

        self.embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
        self.x_embedded = tf.nn.embedding_lookup(self.embeddings, self.X)

        self.loss = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model(), targets=self.Y,
                weights=tf.ones([batch_size, tf.reduce_max(seq_size)], dtype=tf.float32)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, states = tf.nn.dynamic_rnn(
            cell=cell, inputs=self.x_embedded, sequence_length=self.seq_size,
            initial_state=initial_state, dtype=tf.float32, time_major=False)
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        logits = tf.matmul(outputs, self.W) + self.b
        logits = tf.reshape(logits, [self.batch_size, -1, self.dic_size])
        return logits

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            train_x_batch, train_y_batch = tf.train.batch([train_x, train_y], batch_size=self.batch_size)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(2000):
                x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
                _, mse = sess.run([self.train_op, self.loss], feed_dict={self.X: x_batch, self.Y: y_batch})

            coord.request_stop()
            coord.join(threads)
            self.saver.save(sess, './tmp/model.ckpt')
    
    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './tmp/model.ckpt')
            
            output = sess.run(self.model(), feed_dict={self.X: test_x})
            probs = sess.run(tf.nn.softmax(output))
            result = np.argmax(output, axis=2)
            return (probs, result)


