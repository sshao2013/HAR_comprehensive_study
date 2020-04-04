#
# GT-HAR-TOON
#
# CNN-LSTM based HAR
#
# original publication:
# Ordonez and Roggen, Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition Sensors, vol. 16, no. 1, p. 115, Jan. 2016.
#
# to start on command line (and include all own modules):
# export PYTHONPATH="$(find $HOME/Documents/gthartoon -maxdepth 2 -type d | sed '/\/\./d' | tr '\n' ':' | sed 's/:$//')" && python CNN-LSTM_exp.py /data/CBA/datasets/Opportunity/Opp79.mat matlab
#
import lasagne
import theano

import time
import sys

import numpy as np
import theano.tensor as T
import sklearn.metrics as metrics

import messages as M
import data_io
import sliding_win as SW
import theano_lasagne_helper as TLH

######################
# global definitions #
######################
# ** dataset
#DBFNAME= '/Users/thomas/src/AR/lstm_ensembles/data/Opp/Opp79.mat'
DBFNAME= '/Users/thomas/src/AR/lstm_ensembles/data/Skoda.mat'
#DBFNAME= '/Users/thomas/src/AR/lstm_ensembles/data/PAMAP2.mat'
# on hulk2:
#DBFNAME='/data/CBA/datasets/Opportunity/Opp79.mat'
DBFORMAT='MATLAB'

NB_SENSOR_CHANNELS = -1	# will be overridden!

NUM_CLASSES = -1 # will be overridden!

# ** model parameters
# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24
# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12
# Batch Size
BATCH_SIZE = 100
# Number filters convolutional layers
NUM_FILTERS = 64
# Size filters convolutional layers
FILTER_SIZE = 5
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128
# learning rate
LEARNING_RATE = 0.001

NUM_EPOCHS = 100

gt = {}

#########################################################################################
def main():
	global NB_SENSOR_CHANNELS
	global NUM_CLASSES
	global DBFNAME
	global DBFORMAT
	global gt

	start_time = time.time()

	M.msg('**********************')
	M.msg('* CNN-LSTM based HAR *')
	M.msg('**********************')

	M.msg('command line arguments were:')
	for arg in sys.argv[1:]:
		M.msg(arg)
	M.msg('.')

	# first argument has to be path to dataset
	if len(sys.argv) > 1:
		DBFNAME = sys.argv[1]

	# second argument has to be file format
	if len(sys.argv) > 2:
		DBFORMAT = sys.argv[2]

	## read dataset
	data = data_io.read_database(DBFNAME, DBFORMAT)

	# split dataset
	if DBFORMAT.lower() == 'matlab':
		if 'opp' in DBFNAME.lower():	# someone named the data columns differently in opportunity ...
			X_train, X_valid, X_test, y_train, y_valid, y_test = \
				data_io.matlab_unpack(data, True,
							   'trainingData', 'valData', 'testingData',
							   'trainingLabels', 'valLabels', 'testingLabels')
			# Yu's dirty little hack ...
			y_train = y_train - 1
			y_valid = y_valid - 1
			y_test = y_test - 1

			# DEBUG! This is ugly ...
			# Hardcoded number of classes in the gesture recognition problem
			NUM_CLASSES = np.max(y_train) + 1  # 18
		else:	# skoda and pamap2 have consistent naming (data)
			if 'skoda' in DBFNAME.lower() or 'pamap2' in DBFNAME.lower():
				X_train, X_valid, X_test, y_train, y_valid, y_test = \
					data_io.matlab_unpack(data, False,
								   'X_train', 'X_valid', 'X_test',
								   'y_train', 'y_valid', 'y_test')
				# DEBUG! This is ugly ...
				# TODO: find number of classes!
				NUM_CLASSES = np.max(y_train) + 1  # 18 for Opportunity

			#
			# Nothing really works :-( Exit.
			#
			else:
				M.error('Invalid dataset specified', -1)
	else:
		M.error('Only matlab input files supported so far. Sorry.', -1)

	# assign ground truth to some internal variable and store input dimensionality
	gt = y_test.copy()
	NB_SENSOR_CHANNELS = X_test.shape[1]

	## normalise input data (zero mean, unit standard variation)
	M.logmsg(4,'normalising: zero mean, unit variance ...')
	mn_trn = np.mean(X_train, axis=0)
	std_trn = np.std(X_train, axis=0)
	X_train = (X_train - mn_trn) / std_trn
	X_valid = (X_valid - mn_trn) / std_trn
	X_test = (X_test - mn_trn) / std_trn
	M.logmsg(4,'done.')

	## segmentation
	# sliding window "segmentation" here (...)
	M.logmsg(2,'sliding window segmentation ...')
	X_train, X_valid, X_test, y_train, y_valid, y_test = \
			SW.segmentation_slidingwindow(SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS,
										  X_train, X_valid, X_test, y_train, y_valid, y_test)
	M.logmsg(2,'done.')

	## preprocessing
	# no actual preprocessing here


	## configure model
	M.logmsg(2,'configuring CNN model ...')
	net, target_var, cost, updates = _configure_model()
	M.logmsg(2,'done.')

	M.logmsg(2, 'compiling theano functions ...')
	train_fn, test_fn, test_loss_fn = TLH.compile_theano_functions(net, target_var, cost, updates)
	M.logmsg(2,'done.')

	## train / validate / test model
	results = _model_training(NUM_EPOCHS, X_train, y_train, X_valid, y_valid, X_test, y_test, train_fn, test_fn, test_loss_fn)


	## cleanup

	M.msg('*************')
	M.msg('* FINISHED. *')
	M.msg('*************')

	## exit
	quit(0)

#########################################################################################


def _pred_span(preds, span=12):
	pred_all = preds[0] * np.ones(2 * span)
	preds = preds[1:]
	for i in range(len(preds)):
		pred_all = np.concatenate((pred_all, preds[i] * np.ones(span)))

	return pred_all


def _iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


def _model_training(num_epochs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_fn, test_fn, test_loss_fn):
	results = np.zeros(num_epochs)

	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		M.logmsg(3,'starting epoch: ' + str(epoch) + '...')

		train_err = 0
		train_batches = 0
		start_time = time.time()

		M.logmsg(3,'generating minibatches and training on them ...')
		for batch in _iterate_minibatches(X_train, y_train, 100, shuffle=True):
			M.logmsg(4,'... processing minibatch ' + str(train_batches))
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		print("Epoch {} took {:.3f}s with train_loss {:.3f}".format(epoch + 1, time.time() - start_time,
																	train_err / train_batches))
		# print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		sys.stdout.flush()

		# Classification of the testing data
		# print("Processing {0} instances in mini-batches of {1}".format(X_test.shape[0],BATCH_SIZE))

		valid_pred = np.empty((0))
		valid_true = np.empty((0))
		test_pred = np.empty((0))
		test_true = np.empty((0))
		start_time = time.time()

		i=0
		M.logmsg(3,'generating minibatches and testing on them (Validation) ...')
		for batch in _iterate_minibatches(X_valid, y_valid, BATCH_SIZE, shuffle=False):
			M.logmsg(4,'... processing minibatch ' + str(i))
			inputs, targets = batch
			y_pred, = test_fn(inputs)

			valid_pred = np.append(valid_pred, y_pred, axis=0)
			valid_true = np.append(valid_true, targets, axis=0)
			i += 1

		f1_valid = metrics.f1_score(valid_true, valid_pred, average='macro')

		test_err = 0
		test_batches = 0
		M.logmsg(3,'generating minibatches and testing on them (Test) ...')
		for batch in _iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
			M.logmsg(4,'... processing minibatch ' + str(test_batches))
			inputs, targets = batch
			y_pred, = test_fn(inputs)

			test_err += test_loss_fn(inputs, targets)
			test_batches += 1

			test_pred = np.append(test_pred, y_pred, axis=0)
			test_true = np.append(test_true, targets, axis=0)

		f1_test = metrics.f1_score(test_true, test_pred, average='macro')
		print("Epoch {} took {:.3f}s with test_loss {:.3f}, with valid/test f1_scores {:.3f} / {:.3f}".format(epoch + 1,
																											  time.time() - start_time,
																											  test_err / test_batches,
																											  f1_valid,
																											  f1_test))

		test_pred_s = _pred_span(test_pred)
		f1_test_s = metrics.f1_score(gt[:len(test_pred_s)], test_pred_s, average='macro')
		print('------------------------------ sample-wise f1: {:.3f}'.format(f1_test_s))

		if epoch >= 10 and f1_valid > np.max(results):
			print('*********  getting a better model... according to validation data **********')

		# np.save('results/Roggen_'+str(epoch)+'.npy', test_pred)
		# np.savez('model/Roggen_'+str(epoch)+'.npz', *lasagne.layers.get_all_param_values(net['output']))
		results[epoch] = f1_valid

	return results

def _configure_model():
	'''

	:return: net	-- configured lasagne CNN-LSTM
	'''

	##########################################
	#print('layerwised dropout with init!!!')
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	gate_parameters = lasagne.layers.recurrent.Gate(
    						W_in=lasagne.init.Orthogonal(),
    						W_hid=lasagne.init.Orthogonal(),
    						b=lasagne.init.Constant(0.))

	cell_parameters = lasagne.layers.recurrent.Gate(
    						W_in=lasagne.init.Orthogonal(),
    						W_hid=lasagne.init.Orthogonal(),
    						W_cell=None,
							b=lasagne.init.Constant(0.),
    						nonlinearity=lasagne.nonlinearities.tanh)



	net = {}
	net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), input_var)

	net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1), W=lasagne.init.Orthogonal())

	net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=lasagne.init.Orthogonal())


	net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=lasagne.init.Orthogonal())


	net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=lasagne.init.Orthogonal())
	net['conv4/5x1'] = lasagne.layers.DropoutLayer(net['conv4/5x1'], p=0.5)

	net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))

	net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'],
                                        NUM_UNITS_LSTM,
                                        ingate=gate_parameters,
                                        forgetgate=gate_parameters,
                                        cell=cell_parameters,
                                        outgate=gate_parameters)
	net['lstm1'] = lasagne.layers.DropoutLayer(net['lstm1'], p=0.5)

	net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'],
                                        NUM_UNITS_LSTM,
                                        ingate=gate_parameters,
                                        forgetgate=gate_parameters,
                                        cell=cell_parameters,
                                        outgate=gate_parameters)

	#net['lstm2'] = lasagne.layers.DropoutLayer(net['lstm2'], p=0.5)
	# In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
	# to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
	net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))

	#net['drop1'] = lasagne.layers.DropoutLayer(net['shp1'], p=0.5)

	net['prob'] = lasagne.layers.DenseLayer(net['shp1'],NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
	# Tensors reshaped back to the original shape
	net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
	# Last sample in the sequence is considered
	net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)


	network_output = lasagne.layers.get_output(net['output'], deterministic=False) # false for dropout during training

	# The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
	cost = T.nnet.categorical_crossentropy(network_output,target_var).mean()

	# Retrieve all parameters from the network
	all_params = lasagne.layers.get_all_params(net['output'],trainable=True)

	# Compute AdaGrad updates for training
	M.logmsg(4,"Computing updates ...")
	#updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
	updates = lasagne.updates.adam(cost, all_params, LEARNING_RATE)
	M.logmsg(4,'done.')

	return net, target_var, cost, updates



if __name__ == '__main__':
	main()

