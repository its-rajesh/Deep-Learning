#!/usr/bin/env python
# coding: utf-8

# # 3 Layer FCNN with Stochastic Gradient Descent (SGD) Algorithm

# In[9]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import cv2


# In[10]:


import tensorflow as tf
import numpy as np


# ### Reading data as image and storing it into training, validation and testing

# In[186]:


def read_data(path):
    
    train_path = path+"/train"
    test_path = path+"/test"
    validation_path = path+"/val"
    
    tr_data, test_data, val_data = [], [], []
    tr_out, test_out, val_out = [], [], []

    for i in os.listdir(train_path):

        if i != ".DS_Store":
            for j in os.listdir(train_path+"/"+i):
                tr_data.append(cv2.imread(train_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                tr_out.append(i)

            for j in os.listdir(test_path+"/"+i):
                test_data.append(cv2.imread(test_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                test_out.append(i)

            for j in os.listdir(validation_path+"/"+i):
                val_data.append(cv2.imread(validation_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                val_out.append(i)
                
                
    tr_data, test_data, val_data = np.array(tr_data), np.array(test_data), np.array(val_data)
    tr_out, test_out, val_out = np.array(list(map(int, tr_out))), np.array(list(map(int, test_out))), np.array(list(map(int, val_out)))

    return tr_data, test_data, val_data, tr_out, test_out, val_out


# In[223]:


path = "/Users/rajeshr/Desktop/Assignment2/Group_22"
tr_data, test_data, val_data, tr_out, test_out, val_out = read_data(path)


# # SGD

# ### Training

# In[283]:


# Generating Three Layer FCNN Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='Input_Layer'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(10, activation='softmax', name='Output_Layer')
])

# Printing the architecture details of the model
model.summary()

#Setting the optimizer and compiling
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0,name='SGD')
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Setting the convergence criteria and fitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=3)
trained = model.fit(tr_data, tr_out, epochs=10000, batch_size=1, callbacks=callback)

#Plotting error vs epoch graph
plt.plot(trained.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.savefig("/Users/rajeshr/Desktop/"+str(np.random.randint(1, 100000))+".png")
plt.show()


# ### Testing

# In[284]:


loss, mse = model.evaluate(val_data, val_out)


# In[286]:


loss, mse = model.evaluate(test_data, test_out)


# In[285]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# # Batch Mode

# ### Training

# In[287]:


# Generating Three Layer FCNN Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='Input_Layer'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(256, activation='sigmoid', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(10, activation='softmax', name='Output_Layer')
])

# Printing the architecture details of the model
model.summary()

#Setting the optimizer and compiling
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0,name='SGD')
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Setting the convergence criteria and fitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=3)
trained = model.fit(tr_data, tr_out, epochs=10000, batch_size=784, callbacks=callback)

#Plotting error vs epoch graph
plt.plot(trained.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.savefig("/Users/rajeshr/Desktop/"+str(np.random.randint(1, 100000))+".png")
plt.show()


# ### Testing

# In[288]:


loss, mse = model.evaluate(val_data, val_out)


# In[289]:


loss, mse = model.evaluate(test_data, test_out)


# In[290]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# # NAG

# ### Training

# In[291]:


# Generating Three Layer FCNN Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='Input_Layer'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(256, activation='sigmoid', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(10, activation='softmax', name='Output_Layer')
])

# Printing the architecture details of the model
model.summary()

#Setting the optimizer and compiling
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,name='SGD')
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Setting the convergence criteria and fitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=3)
trained = model.fit(tr_data, tr_out, epochs=10000, callbacks=callback)

#Plotting error vs epoch graph
plt.plot(trained.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.savefig("/Users/rajeshr/Desktop/"+str(np.random.randint(1, 100000))+".png")
plt.show()


# ### Testing

# In[292]:


loss, mse = model.evaluate(val_data, val_out)


# In[293]:


loss, mse = model.evaluate(test_data, test_out)


# In[294]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# # RMSProp

# ### Training

# In[299]:


# Generating Three Layer FCNN Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='Input_Layer'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(256, activation='sigmoid', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(512, activation='sigmoid', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(10, activation='softmax', name='Output_Layer')
])

# Printing the architecture details of the model
model.summary()

#Setting the optimizer and compiling
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.99,
    momentum=0.9,
    epsilon=1e-08,
    centered=False,
    name='RMSprop')
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Setting the convergence criteria and fitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=3)
trained = model.fit(tr_data, tr_out, epochs=10000, callbacks=callback)

#Plotting error vs epoch graph
plt.plot(trained.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.savefig("/Users/rajeshr/Desktop/"+str(np.random.randint(1, 100000))+".png")
plt.show()


# ### Testing

# In[300]:


loss, mse = model.evaluate(val_data, val_out)


# In[301]:


loss, mse = model.evaluate(test_data, test_out)


# In[302]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# # Adam Optimizer

# ### Training

# In[303]:


# Generating Three Layer FCNN Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='Input_Layer'),
    tf.keras.layers.Dense(128, activation='sigmoid', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(100, activation='sigmoid', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(128, activation='sigmoid', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(10, activation='softmax', name='Output_Layer')
])

# Printing the architecture details of the model
model.summary()

#Setting the optimizer and compiling
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    name='Adam')
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Setting the convergence criteria and fitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=3)
trained = model.fit(tr_data, tr_out, epochs=10000, callbacks=callback)

#Plotting error vs epoch graph
plt.plot(trained.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.savefig("/Users/rajeshr/Desktop/"+str(np.random.randint(1, 100000))+".png")
plt.show()


# ### Testing

# In[304]:


loss, mse = model.evaluate(val_data, val_out)


# In[305]:


loss, mse = model.evaluate(test_data, test_out)


# In[306]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# ### Confusion Matrix

# In[162]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)


# In[163]:


p_class


# In[165]:


cm = tf.math.confusion_matrix(test_out, p_class)


# In[166]:


cm


# In[ ]:


predictions = model.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(test_out, p_class)
cm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




