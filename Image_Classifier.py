#!/usr/bin/env python
# coding: utf-8

# ## 1. Install Dependencies & Loading Dataset

# In[689]:


import tensorflow as tf
import os


# In[690]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[691]:


import cv2
import imghdr


# In[692]:


data_dir = 'dataset' 


# In[693]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[694]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
#             os.remove(image_path)


# In[695]:


import numpy as np
from matplotlib import pyplot as plt


# In[706]:


data = tf.keras.utils.image_dataset_from_directory('dataset')


# In[707]:


data_iterator = data.as_numpy_iterator()


# In[708]:


batch = data_iterator.next()


# In[709]:


batch[1]


# In[710]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# ## 2. Preprocess Data

# #### Data is now load so lets preprocess image data such that 'image values' scale from 0 to 1 instead of 0 to 255. This helps the DL model during training thus improves results. We will also split up data into various partiitions in order to prevent overfitting.

# In[711]:


data = data.map(lambda x,y: (x/255, y))


# In[712]:


scaled_iterator = data.as_numpy_iterator()


# In[713]:


batch = scaled_iterator.next()


# In[714]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# In[715]:


batch[0]


# In[716]:


train_size = int(len(data)*.7)-1   #  7 * 32 images allocated to training partition
val_size = int(len(data)*.2)+2     #  3 * 32 images allocated to validation partition
test_size = int(len(data)*.1)+1    #  2 * 32 images allocated to test partition


# In[717]:


val_size+train_size+test_size


# In[718]:


# skip and take methods
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# ## Build Model

# In[719]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[720]:


model = Sequential()


# In[721]:


# Archiectural decisions

# Convolution - filter, shape is 3x3, stride (pixel by pixel), relu activation allows us to take care of non linear patterns.
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[722]:


# Adam optimizer
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[727]:


model.summary()


# ### Train the model

# In[728]:


logdir='logs'


# In[729]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) 


# In[730]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[576]:


hist.history


# In[577]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[578]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# ### Performance

# In[579]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[580]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[581]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[582]:


print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# ## Testing!

# In[583]:


import cv2


# In[584]:


img = cv2.imread('shoetest.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


# In[585]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[586]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[587]:


yhat


# In[588]:


if yhat > 0.5: 
    print('Shoe')
else:
    print('Jeans')


# ## Save the model!!!!

# In[443]:


from tensorflow.keras.models import load_model


# In[444]:


# h5 is a serialization of the model similar to zip or rar!
model.save(os.path.join('models','imageclassifier.h5'))


# In[445]:


new_model = load_model(os.path.join('models','imageclassifier.h5'))


# In[446]:


yhat_new = new_model.predict(np.expand_dims(resize/255, 0))


# In[447]:


if yhat_new > 0.5: 
    print(f'Predicted class is Shoe')
else:
    print(f'Predicted class is Jeans')


# In[ ]:




