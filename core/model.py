import os
import math
import numpy as np
import pandas as pd
import datetime as dt
from numpy import newaxis
from core.utils import Timer
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
def StochRSI_EMA(series, period=14, smoothK=3, smoothD=3):
	# Calculate RSI 
	delta = series.diff().dropna()
	ups = delta * 0
	downs = ups.copy()
	ups[delta > 0] = delta[delta > 0]
	downs[delta < 0] = -delta[delta < 0]
	ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
	ups = ups.drop(ups.index[:(period-1)])
	downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
	downs = downs.drop(downs.index[:(period-1)])
	rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
		downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
	rsi = 100 - 100 / (1 + rs)

	# Calculate StochRSI 
	stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
	#stochrsi_K = stochrsi.ewm(span=smoothK).mean()
	#stochrsi_D = stochrsi_K.ewm(span=smoothD).mean()

	return stochrsi
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out
class Model():
	"""A class for an building and inferencing an lstm model"""
	
	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'Flatten':
				self.model.add(Flatten())
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()

	def train(self, x, y, epochs, batch_size, save_dir,X_test,Y_test,saveName):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		#logdir = "logs/scalars/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		
		es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

		tensorboard_callback = TensorBoard(log_dir="logs/scalars/" + saveName+"v3" ,histogram_freq=1)
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs )+"testv3"))
		callbacks = [
			es,
			#EarlyStopping(monitor='val_loss', patience=25),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
			tensorboard_callback
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks,
			validation_data=(X_test, Y_test)
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

		logdir = "logs/scalars/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

		tensorboard_callback = TensorBoard(log_dir=".logs")
		#es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2,restore_best_weights=True)
		es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
			tensorboard_callback,
			es
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()
		

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequences_multipleNewworksih(self, data, ytrain):
		print('[Model] Predicting Sequences Multiple...new')
		y_pred = []
		curr_frame = data
		for j in range(len(data)):
			y_pred.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [len(data)-2], y_pred[-1], axis=0)
		y_train = ytrain
		plt.plot(np.arange(len(y_train), len(y_train) + len(y_train)), y_train[:,0], marker='.', label="true")
		plt.plot(np.arange(len(y_train), len(y_train) + len(y_train)), y_pred, 'r', label="prediction")
		plt.ylabel('Value')
		plt.xlabel('Time Step')
		plt.legend()
		plt.show()

	def predict_sequences_multipleNew(self, data, ytrain):
			print('[Model] Predicting Sequences Multiple...new')
			y_pred = []
			curr_frame = data
			y_pred.append(data[:,0])
			for j in range(len(data)):
				y_pred.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = np.insert(curr_frame,[len(data)-2],y_pred[-1], axis = 0)
			y_pred = curr_frame
			y_train = ytrain
			plt.plot(np.arange(len(y_train), len(y_train) + len(y_train)), y_train[:,0], marker='.', label="true")
			plt.plot(np.arange(len(y_train), len(y_train) + len(y_train)), y_pred, 'r', label="prediction")
			plt.ylabel('Value')
			plt.xlabel('Time Step')
			plt.legend()
			plt.show()
	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
	def predict_sequences_multipleSecondAttempt(self, data, window_size):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Next 2 days...')#MAKE THIS SOMEHOW PREDICT THE ENTIRE 7 DATA POINTS NOT JUST PRICE
		prediction_seqs = []
		curr_frame = data[-29:]
		predicted = []
		
		for j in range(window_size+15):#make the predictions also attatch rsi and shit from previous guesses because its based off the price
			#so you dont need to predict the rsi and stuff because it will be wrong
			#just make a function that outputs it and then append to the 0-6 dimention
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			#curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			#price  = ewma_vectorized(curr_frame[:,0],0.5,8)
			
			df = pd.DataFrame(data = curr_frame)
			ewma8  = df.ewm(span=8,min_periods=0,adjust=False,ignore_na=False).mean().iloc[:,0]
			ewma13 = df.ewm(span=13,min_periods=0,adjust=False,ignore_na=False).mean().iloc[:,0]
			ewma21 = df.ewm(span=21,min_periods=0,adjust=False,ignore_na=False).mean().iloc[:,0]
			ewma55 = df.ewm(span=55,min_periods=0,adjust=False,ignore_na=False).mean().iloc[:,0]
			rsi = StochRSI_EMA(df.iloc[:,0])
			#print(ewma8.iloc[27],"loc")
			#print(rsi)
			a = [predicted[-1],ewma8[27],ewma13[27],ewma21[27],ewma55[27],rsi[27]]
			
			#curr_frame = np.append(curr_frame,a,axis = 0)
			#ar = np.asarray(a)
			curr_frame = np.vstack((curr_frame,a))
			
			#print(predicted[-1])
			
			
		print(curr_frame)
		prediction_seqs.append(predicted)
		return prediction_seqs