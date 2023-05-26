import tensorflow as tf
import numpy as np

class cross_search():
    def __init__(self, model, learning_rate):
        self.model=model
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError())

    def fit(self, X_train, y_train,X_test, y_test, val_split, epochs):
        self.loss =[]
        self.models=[]
        self.X_train = X_train
        self.y_train = y_train
        self.val_split = val_split
        self.epochs=epochs
        for i in range(len(val_split)):
            for j in range(len(epochs)):
                self.model.fit(X_train, y_train, validation_split=val_split[i], batch_size=32, epochs=epochs[j])
                self.loss.append(self.model.evaluate(X_test, y_test, verbose=2))
                self.models.append(self.model)
        self.argminval=np.argmin(self.loss)
        self.model=self.models[self.argminval]
        print(self.loss)
        self.loss=np.array(self.loss)
        self.loss=np.reshape(self.loss,(len(val_split), len(epochs)))
        self.minindex = np.argmin(self.loss,axis=1)
        print('minimal loss for model parameters: ', val_split[self.minindex[0]], epochs[self.minindex[1]])
        print(self.loss)

    def get_best_model(self):
        self.model.fit(self.X_train, self.y_train, validation_split=self.val_split[self.minindex[0]], batch_size=32, epochs=self.epochs[self.minindex[1]])

