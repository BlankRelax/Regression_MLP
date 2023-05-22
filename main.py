import sklearn.metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from keras_sequential_ascii import keras2ascii

###########################Data Wrangling######################################
pd.set_option('display.max_columns', None)
df = pd.read_csv(r'C:\Users\Hassaan\OneDrive\SPA7033U\CW2\life_exp\Life_Expectancy_Data.csv') # change directory here if you want to run it for yourself

df = df.drop(columns=['Country', 'Year'])
df['Status']=pd.Categorical(df['Status'])
df['Status']=df['Status'].cat.codes

columns = df.columns
for var in columns:
    # plt.scatter(df[var], df['Life expectancy '])
    # plt.xlabel(var)
    # plt.ylabel('Life exp')
    # plt.show()
    print("var ", var, 'Life exp' ,np.cov([df[var],df['Life expectancy ']]))

df = df.drop(columns=[' thinness  1-19 years',
                 'Total expenditure',  ' thinness 5-9 years', 'Diphtheria ',  'Population'])
columns = df.columns
df = (df - df.min()) / (df.max() - df.min())
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
imputer = imputer.fit(df)
df = imputer.transform(df)
df = pd.DataFrame(df, columns=columns)
y_df = df['Life expectancy ']
X_df = df.drop(columns=['Life expectancy '])
print(df.columns)
columns=X_df.columns
print(df.loc[:4, :])




X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



#########################Regression Analysis####################################################################



class NN_seq:

    def __init__(self, nodes, activation, input):
        self.model = tf.keras.models.Sequential()
        self.dropoutval = 0.6
        if activation == 'LR':
            self.model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(input, )))

    def add_dense(self, nodes_layers, activation):
            if activation == 'LR':
                i=1
                for num in nodes_layers:
                    if i%2==0 and num>64:
                        self.model.add(tf.keras.layers.Dropout(self.dropoutval))
                    self.model.add(tf.keras.layers.Dense(num, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
                    i+=1
    def add_output_layer(self, nodes):
        self.model.add(tf.keras.layers.Dense(nodes))



m = NN_seq(1028, 'LR', 14)
m.add_dense([1028,1028,512,512,256, 256,128,128,64,64,32,16], 'LR')
m.add_output_layer(1)

model = m.model
print(keras2ascii(model))
loss_func = tf.keras.losses.MeanSquaredError()
learning_rate=0.0001
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=5, restore_best_weights=True)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=loss_func)
history  = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=200)

loss = model.evaluate(X_test,  y_test, verbose=2)
print("loss = {:5.3f}".format(loss))
X_test = pd.DataFrame(data=X_test, columns=columns)
y_predict = model.predict(X_test)
X_test['y_predict'] = y_predict
print(X_test['Adult Mortality'][:10])
print(X_test['y_predict'][:10])
plt.scatter(X_test['Adult Mortality'], X_test['y_predict'], label='Prediction')
plt.scatter(X_test['Adult Mortality'], y_test, label='True class')
plt.ylabel("Life Expectancy")
plt.xlabel("Adult Morality")
plt.legend()
plt.title("Model Prediction vs Truth Values")
plt.show()

delta = []
deltapc = []
for i in range(X_test['Adult Mortality'].shape[0]):
    thedelta = X_test['y_predict'][i]-y_test[i]
    delta.append( thedelta )
    if( X_test['Adult Mortality'][i] ):
       deltapc.append( thedelta /  X_test['Adult Mortality'][i] )
    else:
       deltapc.append( 0.0 )

plt.scatter(X_test['Adult Mortality'], delta)
plt.scatter(X_test['Adult Mortality'], deltapc)
plt.legend(['$\Delta_y$', '$\Delta_y$ (fraction)'], loc='upper right')
plt.title('model prediction accuracy')
plt.ylabel('$\widehat{y}-y$')
plt.xlabel('$x$')
plt.show()
# summarize history for loss
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()
plt.clf()

