import numpy as np
import helpers
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops


class SentenceTokenizer:
    def __init__(self, config, embeddings):
        self.batch_size = config.batch_size
        self.seq_size = config.seq_size
        self.hidden_size = config.hidden_size
    
        self.X = tf.placeholder(tf.int32, [None, self.seq_size])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_size])
        self.W = tf.get_variable('W', dtype=tf.float64, shape=[self.hidden_size, 2], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b', dtype=tf.float64, shape=[2], initializer=tf.contrib.layers.xavier_initializer())

        self.x_embedded = tf.nn.embedding_lookup(embeddings, self.X)

        self.loss = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model(), targets=self.Y,
                weights=tf.ones([self.batch_size, self.seq_size], dtype=tf.float64)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        
        self.filenames = array_ops.placeholder(dtypes.string, shape=[None])
        self.num_epochs = array_ops.placeholder(dtypes.int64, shape=[])
        self.num_batch = array_ops.placeholder(dtypes.int64, shape=[])

        repeat_dataset = dataset_ops.TextLineDataset(self.filenames).repeat(self.num_epochs)
        batch_dataset = repeat_dataset.batch(self.num_batch)

        iterator = dataset_ops.Iterator.from_structure(batch_dataset.output_types)
        self.init_batch_op = iterator.make_initializer(batch_dataset)
        self.get_next = iterator.get_next()
        
        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(
                                    cell=cell, 
                                    inputs=self.x_embedded, 
                                    dtype=tf.float64, 
                                    time_major=False)
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        logits = tf.matmul(outputs, self.W) + self.b
        logits = tf.reshape(logits, [-1, self.seq_size, 2])
        return logits

    def train(self, files, epochs, num_data):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_batch_op, 
                                feed_dict={self.filenames: files, 
                                           self.num_epochs: epochs, 
                                           self.num_batch: self.batch_size*2})
            
            for epoch in range(epochs):
                total_batch = int(num_data / self.batch_size)
                for i in range(total_batch):
                    x_train = []
                    y_train = []
                    file_next = sess.run(self.get_next)
                    for i in range(0, self.batch_size*2, 2):
                        x_train.append(file_next[i].split())
                        y_train.append(file_next[i+1].split())

                    x_train, _ = helpers.batch(x_train, self.seq_size)
                    y_train, _ = helpers.batch(y_train, self.seq_size)
                    _, mse = sess.run([self.train_op, self.loss], feed_dict={self.X: x_train, self.Y: y_train})
            self.saver.save(sess, './tmp/model.ckpt')
    
    def test(self, files, num_data):
        f = open('./result.txt', 'w')
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './tmp/model.ckpt')
            sess.run(self.init_batch_op, 
                                feed_dict={self.filenames: files, self.num_epochs: 1, self.num_batch: 2})
            
            total_acc = 0.
            for i in range(num_data):
                file_next = sess.run(self.get_next)
                x_test = [file_next[0].split()]
                y_test = [file_next[1].split()]
                x_test, _ = helpers.batch(x_test, self.seq_size)
                y_test, _ = helpers.batch(y_test, self.seq_size)
                
                output = sess.run(self.model(), feed_dict={self.X: x_test})
                result = np.argmax(output, axis=2)
                #print('RESULT: ', result)
                #print('ANSWER: ', y_test)
                if np.all(y_test == result):
                    total_acc += 1
                else:
                    f.write('\nY_test: ')
                    print(y_test[0])
                    f.write(str(y_test[0]))
                    f.write('\nResult: ')
                    f.write(str(result[0]))
                if i % 30 == 0:
                    f.write('\nTotal Accuracy: '+str(total_acc))
            f.close() 
            total_acc /= num_data
            total_acc *= 100
            return total_acc
