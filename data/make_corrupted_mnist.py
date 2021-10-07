import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image
from utils import resize_and_crop
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

###############################################################################################
# Code here is partially borrowed from Abubakar Abid: https://github.com/abidlabs/contrastive #
###############################################################################################

SUPERPOSITION_FRACTION_DIGIT = 0.25

# Read MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Only take zeros and ones
target_idx = np.where(Y_train < 2)[0]

# Take first 5000 images
foreground = X_train[target_idx, :][:5000]
foreground = np.array([np.ndarray.flatten(x) for x in foreground])
foreground = foreground / 255
target_labels = Y_train[target_idx][:5000]

mnist_test_digits = X_train[target_idx, :][5000:10000]
mnist_test_digits = np.array([np.ndarray.flatten(x) for x in mnist_test_digits])
mnist_test_digits = mnist_test_digits / 255

np.save("./corrupted_mnist/mnist_digits_test.npy", mnist_test_digits)

num_zeros = np.where(target_labels == 0)[0].shape[0]
num_ones = np.where(target_labels == 1)[0].shape[0]
print("Loaded {} zeros and {} ones".format(num_zeros, num_ones))

# Load grass images

print("Loading grass images...")
IMAGE_PATH = "./grass"  # Replace with your own path to downloaded images

natural_images = (
    list()
)  # dictionary of pictures indexed by the pic # and each value is 100x100 image
for filename in tqdm(os.listdir(IMAGE_PATH)):
    if (
        filename.endswith(".JPEG")
        or filename.endswith(".JPG")
        or filename.endswith(".jpg")
    ):
        im = Image.open(os.path.join(IMAGE_PATH, filename))
        im = im.convert(mode="L")  # convert to grayscale
        im = resize_and_crop(im)  # resize and crop each picture to be 100px by 100px
        natural_images.append(np.reshape(im, [10000]))

natural_images = np.asarray(natural_images, dtype=float)
natural_images /= 255  # rescale to be 0-1
print("Array of grass images:", natural_images.shape)


# Corrupt the MNIST digits with the grass
np.random.seed(0)  # for reproducibility

rand_indices = np.random.permutation(
    natural_images.shape[0]
)  # just shuffles the indices
split = int(len(rand_indices) / 2)
target_indices = rand_indices[
    0:split
]  # choose the first half of images to be superimposed on target
background_indices = rand_indices[
    split:
]  # choose the second half of images to be background dataset

target = np.zeros(foreground.shape)
background = np.zeros(foreground.shape)

print("Corrupting MNIST with grass...")
for i in tqdm(range(target.shape[0])):

    # Foreground
    # (MNIST corrupted with grass)
    idx = np.random.choice(target_indices)  # randomly pick a image
    loc = np.random.randint(70, size=(2))  # randomly pick a region in the image

    reshaped_grass = np.reshape(natural_images[idx, :], [100, 100])
    x1, x2, y1, y2 = loc[0], loc[0] + 28, loc[1], loc[1] + 28
    grass_cropped = reshaped_grass[x1:x2, :][:, y1:y2]
    superimposed_patch = np.reshape(grass_cropped, [1, 784])
    target[i] = SUPERPOSITION_FRACTION_DIGIT * foreground[i] + superimposed_patch

    # Background
    # (Just grass)
    idx = np.random.choice(background_indices)  # randomly pick a image
    loc = np.random.randint(70, size=(2))  # randomly pick a region in the image

    reshaped_grass = np.reshape(natural_images[idx, :], [100, 100])
    x1, x2, y1, y2 = loc[0], loc[0] + 28, loc[1], loc[1] + 28
    grass_cropped = reshaped_grass[x1:x2, :][:, y1:y2]

    background_patch = np.reshape(grass_cropped, [1, 784])
    background[i] = background_patch

np.save("./corrupted_mnist/foreground.npy", target)
np.save("./corrupted_mnist/background.npy", background)
np.save("./corrupted_mnist/foreground_labels.npy", target_labels)

# logreg = LogisticRegression()
# logreg.fit(target[:4000], target_labels[:4000])
# train_acc = logreg.score(target[4000:], target_labels[4000:])
# print("Train accuracy: {}".format(round(train_acc, 3)))

n_show = 6

plt.figure(figsize=[21, 7])
for i in range(n_show):
    plt.subplot(2, n_show, i + 1)
    idx = np.random.randint(5000)
    plt.imshow(
        np.reshape(target[idx, :], [28, 28]), cmap="gray", interpolation="bicubic"
    )
    plt.axis("off")

for i in range(n_show):
    plt.subplot(2, n_show, n_show + i + 1)
    idx = np.random.randint(5000)
    plt.imshow(
        np.reshape(background[idx, :], [28, 28]), cmap="gray", interpolation="bicubic"
    )
    plt.axis("off")

plt.savefig("../plots/corrupted_mnist/corrupted_mnist_example.png")
plt.show()
# import ipdb; ipdb.set_trace()
