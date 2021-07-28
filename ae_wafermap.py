# (C) 2019 Griffin Bishop - http://griffinbishop.com #
# Video in action: https://gfycat.com/inconsequentialesteemeddartfrog
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os



bottleneck_size = 2

input_img = Input(shape=(64*64,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(bottleneck_size, activation='linear')(encoded)
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(bottleneck_size,))
decoded = Dense(64, activation='relu')(encoded_input)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(64*64, activation='sigmoid')(decoded)
decoder = Model(encoded_input, decoded)

full = decoder(encoder(input_img))
ae = Model(input_img, full)
ae.compile(optimizer='adam', loss='mean_squared_error')


######
import cv2
import numpy as np
import glob
images = glob.glob("training/*.png")

dataset = []

for image in images:
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_NEAREST)
    dataset.append(img)

X = np.array(dataset)
X = X.astype('float32')  / 255.
X = X.reshape(X.shape[0], 64*64)
y = np.zeros((X.shape[0], ))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=25)


###########
if "model.h5" not in os.listdir():
    ae = load_model('model.h5')
    encoder = load_model('encoder.h5')
    decoder = load_model('decoder.h5')
else:
    ae.fit(x_train, x_train, 
        epochs = 500,
        batch_size=4,
        validation_data=(x_test, x_test))
    ae.save('model.h5')
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')


encoded_imgs = encoder.predict(x_train)
check_dict = {}
for i, e_img in enumerate(encoded_imgs):
    key = str(int(e_img[0]))+str(int(e_img[1]))
    if key not in check_dict:
        check_dict[key] = x_train[i]

check_list = list(check_dict.keys())

_ = encoded_imgs.astype(int)
decoded_imgs = decoder.predict(encoded_imgs)




fig, ax = plt.subplots(1, 3)
for i in range(len(encoded_imgs[:,0])):
    ax[0].plot(encoded_imgs[i, 0], encoded_imgs[i, 1], 'o', picker=True, pickradius=3)


def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    ix = xdata[ind]
    iy = ydata[ind]

    latent_vector = np.array([[ix, iy]])
    decoded_img = decoder.predict(latent_vector)
    decoded_img = decoded_img.reshape(64, 64)*255.
    decoded_img = cv2.resize(decoded_img, (28, 28), interpolation = cv2.INTER_NEAREST)

    ax[1].imshow(decoded_img, cmap='gray')
    if str(int(ix))+str(int(iy)) in check_list:
        o_img = check_dict[str(int(ix))+str(int(iy))]
        o_img = o_img.reshape(64, 64)*255.
        o_img = cv2.resize(o_img, (28, 28), interpolation = cv2.INTER_NEAREST)
        ax[2].imshow(o_img, cmap='gray')
    plt.draw()

# button_press_event
fig.canvas.mpl_connect('pick_event', onpick)

plt.show() 


