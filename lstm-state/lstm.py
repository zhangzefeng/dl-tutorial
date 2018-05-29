from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from numpy import array

# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape((2, 3, 1))
# make and show prediction
print(model.predict(data))
