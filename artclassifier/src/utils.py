import logging
import os
from datetime import datetime
import cv2
import numpy as np
from tensorflow.keras.applications import vgg16, mobilenet_v2, xception


def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    # cv2.putText()
    return NotImplementedError


def predictor(image,model): 
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    
    # reshape image to (1, 224, 224, 3)
    image = image.reshape((1, 224, 224, 3))
    
    # apply pre-processing
    image = mobilenet_v2.preprocess_input(image)

    # make the prediction
    y_pred = model.predict(image).round(2)
    y_pred = y_pred.round(2)

    print('This art belongs to:')
    print(f' Expressionism:      {y_pred[0][0]*100}%')
    print(f' Impressionism:       {y_pred[0][1]*100}%')
    print(f' Cubism: {y_pred[0][2]*100}%')
    print(f' Pop Art:    {y_pred[0][3]*100}%')


def predict_frame(image): 
    # (image, mobilenet_v2, MobileNetV2)
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    
    # reshape image to (1, 224, 224, 3)
    image = image.reshape((1, 224, 224, 3))
    
    # apply pre-processing
    image = mobilenet_v2.preprocess_input(image)

    # define the model
    model = mobilenet_v2.MobileNetV2()

    # make the prediction
    y_pred = model.predict(image)
    
    print(mobilenet_v2.decode_predictions(y_pred, top=5))