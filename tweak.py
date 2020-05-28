#Tweaking the model if got less accuracy

file = open("tweak.txt","r")
a=file.read()
if a=="0":
	#importing all the required libraries

	import keras
	from keras.datasets import fashion_mnist 
	from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
	from keras.models import Sequential
	from keras.utils import to_categorical
	import numpy as np
	import matplotlib.pyplot as plt

	#loading the mnist dataset

	data = np.load('/root/mnist.npz')

	x_train = data['x_train']
	y_train = data['y_train']
	x_test = data['x_test']
	y_test = data['y_test']


	x_train = x_train.reshape(-1, 28,28, 1)
	x_test = x_test.reshape(-1, 28,28, 1)	

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255
	x_test = x_test / 255


	train_Y_one_hot = to_categorical(y_train)
	test_Y_one_hot = to_categorical(y_test)
	
	#creating the model

	model = Sequential()
	
	#adding the layers and initilising the hyperparameters
	
	
	model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(1,1)))
	model.add(Conv2D(64, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.04),metrics=['accuracy'])

	#fitting the data

	model.fit(x_train, train_Y_one_hot, batch_size=64, epochs=3)

	#gaining the accuracy

	test_loss, test_acc = model.evaluate(x_test, test_Y_one_hot)


	print("You got the excellent accuracy ",test_acc)

	


	
else:
	pass
