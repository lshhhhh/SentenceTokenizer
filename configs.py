class SmallConfig(object):
    epochs = 10
    learning_rate = 0.01
    vocab_size = 8000
    hidden_size = 64
    seq_size = 5
    batch_size = 32
    embedding_size = 64


class TestConfig(object):
    '''Config for testing simple data.'''
    epochs = 1000
    learning_rate = 0.1
    vocab_size = 8000
    hidden_size = 256
    seq_size = 5
    batch_size = 16
    embedding_size = 64
