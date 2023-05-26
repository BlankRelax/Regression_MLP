import tensorflow as tf
import numpy as np

class cross_search():
    def __init__(self, model, learning_rate):
        self.model=model
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError())

    def fit(self, X_train, y_train,X_test, y_test, val_split, epochs):
        self.loss ={}
        self.models=[]
        for i in range(len(val_split)):
            for j in range(len(epochs)):
                self.model.fit(X_train, y_train, validation_split=val_split[i], batch_size=32, epochs=epochs[j])
                self.loss.append(self.model.evaluate(X_test, y_test, verbose=2))
                self.models.append(self.model)
        self.model = self.models[np.argmin(self.loss)]
        print('minimal loss for model parameters: ', val_split[np.argmin(self.loss)], epochs[np.argmin(self.loss)])
