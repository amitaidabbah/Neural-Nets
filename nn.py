############# IMPORTS ##################################################################################################
import random
from math import ceil
import numpy as np
from imageio import imread
from skimage.color import rgb2grey
from skimage.util import view_as_windows
from scipy.ndimage.filters import convolve
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Reshape, Flatten
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from . import sol5_utils

########################################################################################################################

############# CONSTANTS ################################################################################################

GRAYSCALE = 1
RGB = 2

########################################################################################################################


############# HELPER FUNCS #############################################################################################

def read_image(filename, representation):
    """
    this function read an image.
    :param filename: image name or path to read.
    :param representation: 1 for RGB, 2 for GRAYSCALE
    :return: an image in float64 format.
    """
    if representation == GRAYSCALE:
        image = imread(filename)
        if image.ndim == 3:
            return rgb2grey(image)
        return image / 255
    elif representation == RGB:
        image = imread(filename)
        return image / 255
    else:
        exit()


def create_patch(random_image, crop_size, corruption_func):
    """
    creates a random patch tuple from input image with the size of (crop_size).
    :param random_image: image to patch from
    :param crop_size: tuple patch size
    :param corruption_func: corruption func to apply
    :return: original patch,corrupt patch
    """
    crop_x, crop_y = crop_size[0], crop_size[1]
    # Generate patches:
    patches = view_as_windows(np.ascontiguousarray(random_image), (3 * crop_x, 3 * crop_y))
    x, y, _, _ = patches.shape
    x, y = random.randint(0, x - 1), random.randint(0, y - 1)
    # Choose random patch:
    random_patch = patches[x, y]

    # Corrupt random patch:
    corrupt_random_patch = corruption_func(random_patch)

    x, y = random.randint(0, crop_x - 1), random.randint(0, crop_y - 1)
    # Create patches from patches and select one randomly:
    random_patch_patches = view_as_windows(np.ascontiguousarray(random_patch), (crop_x, crop_y))
    currupt_random_patch_patches = view_as_windows(np.ascontiguousarray(corrupt_random_patch), (crop_x, crop_y))

    return random_patch_patches[x, y][:, :] - 0.5, currupt_random_patch_patches[x, y][:, :] - 0.5


########################################################################################################################

############# MAIN CODE ################################################################################################
def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    generates random tuples of the form (source_batch, target_batch), where each output variable is an array of shape
    (batch_size, height,width, 1). target_batch is made of clean images, and source_batch is their respective randomly
    corrupted version according to corruption_func(im).
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:
    """
    images_dict = {}
    len(filenames)
    while True:
        for b in range(0, batch_size):
            image_path = random.choice(filenames)

            # Load example:
            if image_path in images_dict:
                random_image = images_dict[image_path]
            else:
                random_image = read_image(image_path, 1)
                images_dict[image_path] = random_image

            # Create patches:
            patch, corrupt_patch = create_patch(random_image, crop_size, corruption_func)

            # Add dimensions so it can be concatenated along this axis:
            patch = np.expand_dims(patch, axis=0)
            patch = np.expand_dims(patch, axis=3)
            corrupt_patch = np.expand_dims(corrupt_patch, axis=0)
            corrupt_patch = np.expand_dims(corrupt_patch, axis=3)

            if b == 0:
                # First example in batch
                batch_patch = patch
                batch_corrupt_patch = corrupt_patch
            else:
                batch_patch = np.concatenate((batch_patch, patch), axis=0)
                batch_corrupt_patch = np.concatenate((batch_corrupt_patch, corrupt_patch), axis=0)
        yield (batch_corrupt_patch, batch_patch)
        batch_patch = None
        batch_corrupt_patch = None


def resblock(input_tensor, num_channels):
    """
    creates a Residual Block.
    :param input_tensor: input tensor for the residual bloch
    :param num_channels: num of channel for the conv layer in the block
    :return: a block.
    """
    a = Conv2D(num_channels, kernel_size=(3, 3), padding='same')(input_tensor)
    b = Activation('relu')(a)
    c = Conv2D(num_channels, kernel_size=(3, 3), padding='same')(b)
    d = Add()([input_tensor, c])
    return d


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    builds a neural network based on residual blocks
    :param height: height of the input
    :param width: width of the input
    :param num_channels: num of channels in conv layers
    :param num_res_blocks: nuber of blocks in network
    :return: keras Model object defining the network.
    """
    input_layer = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, kernel_size=(3, 3), padding='same')(input_layer)
    output_layer = Activation('relu')(b)
    for i in range(num_res_blocks):
        resb = resblock(output_layer, num_channels)
        output_layer = resb
    conv_layer = Conv2D(1, kernel_size=(3, 3), padding='same')(output_layer)
    output_layer = Add()([input_layer, conv_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    """
    this funtion gets a model and data and trains a network based on these inputs.
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and
                   should append anything to them
    :param corruption_func:
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch
    :return:
    """
    num_of_images = len(images)
    training_set = images[0:ceil(0.8 * num_of_images)]
    val_set = images[ceil(0.8 * num_of_images) + 1:]
    training_set_gen = load_dataset(training_set, batch_size, corruption_func, model.input_shape[1:3])
    val_set_gen = load_dataset(val_set, batch_size, corruption_func, model.input_shape[1:3])
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_set_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs,
                        validation_data=val_set_gen, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    restores a corrrupted image.
    :param corrupted_image: the corrupted image to restore
    :param base_model: a model to apply to the corrupted image in order to restore it
    :return: the restored image.
    """
    x, y = corrupted_image.shape[0], corrupted_image.shape[1]
    a = Input((x, y, 1))  # with width and height of the image to be restored.
    b = base_model(a)
    new_model = Model(inputs=[a], outputs=[b])
    corrupted_image = np.expand_dims(corrupted_image, axis=0)
    corrupted_image = np.expand_dims(corrupted_image, axis=3)
    img = new_model.predict(corrupted_image - 0.5)
    img = np.squeeze(img, axis=3)
    img = np.squeeze(img, axis=0)
    return img.astype(np.float64) + 0.5


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds random gaussian noise to an image.
    :param image: image to add noise.
    :param min_sigma: min sigma of noise
    :param max_sigma: max sigma of noise
    :return: noised image
    """
    rand_sigma = random.uniform(min_sigma, max_sigma)
    randon_gaussian_image = np.random.normal(loc=0, scale=rand_sigma, size=image.shape)
    image_with_noise = np.round((randon_gaussian_image + image) * 255) / 255
    image_with_noise = np.clip(image_with_noise, 0, 1)
    return image_with_noise


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    trains a residual convnet to denoise images.
    :param num_res_blocks: num of res block in the network.
    :param quick_mode: for debug only. reduces num of iterations.
    :return: trained model.
    """
    batch_size, steps_per_epoch, epochs, val_samples = 100, 100, 5, 1000
    if quick_mode:
        batch_size, steps_per_epoch, epochs, val_samples = 10, 3, 2, 30
    model = build_nn_model(24, 24, 48, num_res_blocks)
    image_pathes = sol5_utils.images_for_denoising()
    train_model(model, image_pathes, lambda x: add_gaussian_noise(x, 0, 0.2), batch_size, steps_per_epoch, epochs,
                val_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    adds motion blur to an image.
    :param image: image to add blur to
    :param kernel_size: size of kernel to add blur. bigger kernel == more blur
    :param angle: angle of blur.
    :return: blurred image.
    """
    blurr_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    blurred_image = convolve(image, blurr_kernel, mode='constant')
    blurred_image = np.round(blurred_image * 255) / 255
    blurred_image = np.clip(blurred_image, 0, 1)
    return blurred_image


def random_motion_blur(image, list_of_kernel_sizes):
    """
    adds random motion blur to an image
    :param image: image to add motion blur
    :param list_of_kernel_sizes: list of kernel sizes to choose from
    :return: randomly blurred image.
    """
    angle = random.uniform(0, np.pi)
    size = random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, size, angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    trains a residual convnet to deblurr images.
    :param num_res_blocks: num of res block in the network.
    :param quick_mode: for debug only. reduces num of iterations.
    :return: trained model.
    """
    batch_size, steps_per_epoch, epochs, val_samples = 100, 100, 10, 1000
    if quick_mode:
        batch_size, steps_per_epoch, epochs, val_samples = 10, 3, 2, 30
    model = build_nn_model(16, 16, 32, num_res_blocks)
    image_pathes = sol5_utils.images_for_deblurring()
    train_model(model, image_pathes, lambda x: random_motion_blur(x, [7]), batch_size, steps_per_epoch, epochs,
                val_samples)
    return model


########################################################################################################################
############# BONUS ####################################################################################################


encoding_size = 257
perturbation_max = 40

preprocess = lambda x: x / 127 - 1
deprocess = lambda x: ((x + 1) * 127).astype(np.uint8)


def build_decoder():
    """
    builds an encoder-decoder convnet similar to what was implemented in the article 'deep-image-prior'.
    :return:
    """
    model_input = Input(shape=(64, 64, 1))
    conv1 = Conv2D(32, 3, padding='same', activation='relu')(model_input)
    conv2 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    strided_conv1 = Conv2D(32, 3, strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(64, 3, padding='same', activation='relu')(strided_conv1)
    conv4 = Conv2D(64, 3, padding='same', activation='relu')(conv3)
    strided_conv2 = Conv2D(64, 3, strides=(2, 2), padding='same')(conv4)
    conv5 = Conv2D(128, 3, padding='same', activation='relu')(strided_conv2)
    conv6 = Conv2D(128, 3, padding='same', activation='relu')(conv5)
    flatten = Flatten()(conv6)
    encoding = Dense(encoding_size, activation='relu')(flatten)
    dense2 = Dense(48, activation='relu')(encoding)
    reshape = Reshape((4, 4, 3))(dense2)
    upsample2 = UpSampling2D(size=(4, 4))(reshape)
    conv11 = Conv2D(128, 3, padding='same', activation='relu')(upsample2)
    conv12 = Conv2D(128, 3, padding='same', activation='relu')(conv11)
    add1 = Add()([conv12, conv6])
    upsample3 = UpSampling2D()(add1)
    conv13 = Conv2D(64, 3, padding='same', activation='relu')(upsample3)
    conv14 = Conv2D(64, 3, padding='same', activation='relu')(conv13)
    upsample3 = UpSampling2D()(conv14)
    conv15 = Conv2D(8, 3, padding='same', activation='relu')(upsample3)
    conv16 = Conv2D(3, 3, padding='same', activation='tanh')(conv15)

    autoencoder = Model(model_input, conv16)
    autoencoder.compile(Adam(1e-3), loss='mse')
    return autoencoder


def deep_prior_restore_image(corrupted_image):
    """
    restores a noised image using non learning technique implemented in the article 'deep-image-prior'.
    :param corrupted_image: 64X64 RGB (64,64,3) noised image to restore
    :return: denoised image.
    """
    im_shape = corrupted_image.shape
    base_image = np.random.random(size=(1,) + im_shape) * 2 - 1
    base_image = np.expand_dims(base_image, 3)
    corrupted_img_batch = np.expand_dims(corrupted_image, 0)
    corrupted_img_batch = np.expand_dims(corrupted_img_batch, 3)

    FIT_PARAMS = {
        'x': base_image,
        'y': corrupted_img_batch,
        'epochs': 180,
        'batch_size': 1,
        'verbose': 0
    }
    autoencoder = build_decoder()
    autoencoder.fit(**FIT_PARAMS)
    return deprocess(autoencoder.predict(base_image)[0])

########################################################################################################################
