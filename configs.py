class LargeConfig(object):
    '''Large config'''
    epochs = 1
    learning_rate = 0.1
    vocab_size = 8000
    hidden_size = 64
    
    batch_size = 16
    seq_size = 70
    embedding_size = 64


class TestConfig(object):
    '''Tiny config, for testing simple data.'''
    epochs = 1000
    learning_rate = 0.1
    vocab_size = 8000
    hidden_size = 256
    
    batch_size = 16
    seq_size = 70
    embedding_size = 64

