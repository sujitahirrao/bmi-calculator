import cv2
import dlib
import numpy as np
import tensorflow as tf

from src import config
from src.predict_bmi_from_face_image.model import get_model


def get_trained_model():
    weights_file = config.MODEL_WEIGHTS_PATH
    _model = get_model(ignore_age_weights=True)
    _model.load_weights(weights_file)
    return _model


print('Loading model to detect BMI...')
model = get_trained_model()
detector = dlib.get_frontal_face_detector()

graph = tf.get_default_graph()


def predict(image_file_path):
    img = cv2.imread(image_file_path)
    img = cv2.resize(img, (640, 480))
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    detected = detector(input_img, 1)
    faces = np.empty((len(detected), config.RESNET50_DEFAULT_IMG_WIDTH,
                      config.RESNET50_DEFAULT_IMG_WIDTH, 3))

    for i, d in enumerate(detected):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, \
                               d.bottom() + 1, d.width(), d.height()
        xw1 = max(int(x1 - config.MARGIN * w), 0)
        yw1 = max(int(y1 - config.MARGIN * h), 0)
        xw2 = min(int(x2 + config.MARGIN * w), img_w - 1)
        yw2 = min(int(y2 + config.MARGIN * h), img_h - 1)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(''.join(image_file_path.split('.')[:-1]) + '-cropped.jpg',
                    img[yw1:yw2, xw1:xw2])
        faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (
            config.RESNET50_DEFAULT_IMG_WIDTH,
            config.RESNET50_DEFAULT_IMG_WIDTH)) / 255.00

    with graph.as_default():
        predictions = model.predict(faces)

    return predictions[0][0]
