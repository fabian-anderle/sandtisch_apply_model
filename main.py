import cv2
import onnxruntime
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
#from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt

input_img_name = "input.jpg"
width, height = 256, 256
models = {
    "styria_onnx": "sat_model_styria.onnx",
    "tyrol_onnx": "sat_model_tyrol.onnx",
    "styria_h5": "sat_model_styria_300K.h5"
}

def import_img(file_name:str):
    return tf.io.read_file(file_name)

def load_model(model_name:str):
    return tf.keras.models.load_model(model_name), model_name

def load_model_onnx_as_tf(model_name:str):
    onnx_model = onnx.load(model_name)
    return prepare(onnx_model), model_name

def load_model_onnx(model_name:str):
    return onnxruntime.InferenceSession(model_name), model_name

def gen_image_tf(model, input_data):
    return model(input_data, training=True)

def gen_image_onnx(model, input_data):
    return model.run(None, input_data)

def convert_data_tf(img):
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    tensor = tf.image.resize(tensor, [width, height])
    return tf.expand_dims(tensor, axis=0)

def convert_data_onnx(img):
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    tensor = tf.image.resize(tensor, [width, height])
    tensor = tf.expand_dims(tensor, axis=0)
    return [{'x': tensor}]

def show_img(img, title:str, wait_key = 0):
    cv2.imshow(title, img)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()

def plot_images_from_dict(image_dict):
    num_images = len(image_dict)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, (title, image) in enumerate(image_dict.items()):
        axs[i].imshow(image)
        axs[i].set_title(title)
        axs[i].axis('off')

    plt.show()

if __name__ == '__main__':
    img_depth = import_img(input_img_name)
    model, model_used = load_model(models["styria_h5"])
    img_depth_converted = convert_data_tf(img_depth)
    img_sat = gen_image_tf(model, img_depth_converted)

    display_list = [img_depth_converted[0], img_sat[0]]
    title = ['Input Image', 'Output Image']
    display_imgs = {
        'Input Image Processed': img_depth_converted[0],
        'Output Image': img_sat[0]
    }

    plot_images_from_dict(display_imgs)

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()



