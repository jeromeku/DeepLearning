import theano
import theano.tensor as T
from theano import function,shared
from theano.tensor.shared_randomstreams import RandomStreams

import os, sys, gzip, cPickle, time, logging
import numpy as np


data_path = 'DeepLearningTutorials/mnist.pkl.gz'

class DataLoader(object):
    def __init__(self,data_path):
        self.data_path = data_path
    
    def _load_raw_data(self):
        print '... loading data'
        # Load the dataset
        f = gzip.open(self.data_path, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set,valid_set,test_set

    def __createShared(self,data,dtype,borrow):
        return theano.shared(np.asarray(data,dtype=dtype),borrow=borrow)

    def _shared_dataset(self,data_xy,borrow=True):
        data_x,data_y = data_xy
        shared_x = self.__createShared(data_x,dtype=theano.config.floatX,borrow=borrow)
        shared_y = self.__createShared(data_y,dtype=theano.config.floatX,borrow=borrow)
        return shared_x,T.cast(shared_y,'int32')

    def load_data(self):
        train_set,valid_set,test_set = self._load_raw_data()
        print 'Training set dim: {} x {}'.format(train_set[0].shape[0],train_set[0].shape[1])
        print 'Valid set dim: {} x {}'.format(valid_set[0].shape[0],valid_set[0].shape[1])
        print 'Test set dim: {} x {}'.format(test_set[0].shape[0],test_set[0].shape[1])

        test_set_x, test_set_y = self._shared_dataset(test_set)
        valid_set_x, valid_set_y = self._shared_dataset(valid_set)
        train_set_x, train_set_y = self._shared_dataset(train_set)

        return {'Train': (train_set_x,train_set_y),'Validation':(valid_set_x,valid_set_y),\
                'Test': (test_set_x,test_set_y)}

class LogisticRegression(object):
    
    def __init__(self,data_in,n_in,n_out):
        self.W = shared(value=np.zeros((n_in,n_out),dtype=theano.config.floatX),name='W')
        self.b = shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b')

        #Likelihood
        self.p_y_given_x = T.nnet.softmax(T.dot(data_in,self.W)+self.b)
    
        #Predicted class
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
    
    def negative_log_likelihood(self,y):
        row_index = T.arange(y.shape[0])
        log_likelihood = T.log(self.p_y_given_x)[row_index,y]
        return -T.mean(log_likelihood)
    
    def errors(self,y):
         # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
def sgd_optimization(learning_rate=.13,n_epochs=500,dataset='mnist.pkl.gz',batch_size=600):
    #Load and Prep Data
    datasets = DataLoader(data_path).load_data()
    train_set_x,train_set_y = datasets['Train']
    test_set_x,test_set_y = datasets['Test']
    valid_set_x,valid_set_y = datasets['Validation']
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    #Build Models
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    data_feature_dim = train_set_x.get_value(borrow=True).shape[1]

    classifier = LogisticRegression(data_in=x,n_in=data_feature_dim,n_out=10)
    cost = classifier.negative_log_likelihood(y)

    test_model = function(inputs=[index],outputs=classifier.errors(y),givens={x: test_set_x[index*batch_size:(index+1)*batch_size],\
                                                                              y: test_set_y[index*batch_size:(index+1)*batch_size]})

    valid_model = function(inputs=[index],outputs=classifier.errors(y),givens={x: valid_set_x[index*batch_size:(index+1)*batch_size],\
                                                                               y: valid_set_y[index*batch_size:(index+1)*batch_size]})

    g_W = T.grad(cost=cost,wrt=classifier.W)
    g_b = T.grad(cost=cost,wrt=classifier.b)

    updates = [(classifier.W,classifier.W - learning_rate*g_W),(classifier.b,classifier.b - learning_rate*g_b)]

    train_model = function(inputs=[index],outputs=cost,updates=updates,givens={ x: train_set_x[index*batch_size:(index+1)*batch_size],\
                                                                                y: train_set_y[index*batch_size:(index+1)*batch_size]})
    
    #Training Loop
    print "-- Training Model -- "

    ##Early Stopping Criteria
    patience = 5000 #patience is in units of mini-batches 
    patience_increase = 2
    improvement_threshold = .995
    validation_frequency = min(n_train_batches,patience/2)

    best_params = None
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            avg_cost = train_model(minibatch_index)

            num_iter = (epoch - 1)*n_train_batches + minibatch_index


            if((num_iter+1) % validation_frequency == 0):
                #Score Model on Validation Set
                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                current_validation_loss = np.mean(validation_losses)

                logger.info('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches,
                        current_validation_loss * 100.))
                if current_validation_loss < best_validation_loss:
                    if (current_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience,num_iter*patience_increase)

                    best_validation_loss = current_validation_loss

                    #Score Model on Test Set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    logger.info(('     epoch %i, minibatch %i/%i, test error of best'
                           ' model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                             test_score * 100.))
            if patience <= num_iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code ran for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print 'The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))


if __name__ == '__main__':
	log_file = "sgd_test.log"
	logger = logging.getLogger(__file__)
	logger.setLevel(logging.INFO)
	hdlr = logging.FileHandler('test.log')
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.info("About to start training...")
	sgd_optimization()

