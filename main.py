import cv2
import onnxruntime
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from matplotlib import pyplot as plt

output_img_name = "output.jpg"
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

def plot_images_from_dict(image_dict):
    num_images = len(image_dict)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, (title, image) in enumerate(image_dict.items()):
        axs[i].imshow(image * 0.5 + 0.5)
        axs[i].set_title(title)
        axs[i].axis('off')

    plt.show()

def save_output_img(img, name):
    cv2.imwrite(name, img)

def adjust_channels(img):
    img_array = np.array(img * 127.5 + 127.5)
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    img_depth = import_img(input_img_name)
    model, model_used = load_model(models["styria_h5"])
    img_depth_converted = convert_data_tf(img_depth)
    img_sat = gen_image_tf(model, img_depth_converted)

    display_imgs = {
        'Input Image Processed': img_depth_converted[0],
        'Output Image': img_sat[0]
    }

    plot_images_from_dict(display_imgs)
    save_output_img(adjust_channels(img_sat[0]), output_img_name)


