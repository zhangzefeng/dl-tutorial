# try use dense layer directly instead of lstm for attention model
from random import randint
from numpy import array
from numpy import array_equal
from keras.models import Sequential
from keras.layers import Dense

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]

    X = array(sequence_in)
    y = array(sequence_out)
    X = X.reshape((1, X.shape[0]))
    y = y.reshape((1, y.shape[0]))
    return X,y

# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
# define model
model = Sequential()
model.add(Dense(256, input_shape=(n_timesteps_in,)))
model.add(Dense(n_timesteps_in))
model.compile(loss='mean_squared_error', optimizer = 'rmsprop')
# train LSTM
for epoch in range(5000):
    # generate new random sequence
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, verbose=2)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    if array_equal(y[0], yhat[0]):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    print('Expected:', y[0], 'Predicted', yhat[0])
