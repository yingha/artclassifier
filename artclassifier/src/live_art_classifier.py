import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam, predictor
import tensorflow.keras as keras


logging.getLogger().setLevel(logging.INFO)

package_dir = os.path.dirname(__file__)
print(package_dir)

# define the model
model = keras.models.load_model(package_dir +'/models/art_styles_model.h5')

# also try out this resolution: 640 x 360
webcam = init_cam(640, 480)
key = None

try:
    # q key not pressed 
    while key != 'q':
        # Capture frame-by-frame
        ret, frame = webcam.read()
        # fliping the image 
        frame = cv2.flip(frame, 1)
   
        # draw a [224x224] rectangle into the frame, leave some space for the black border 
        offset = 2
        width = 224
        x = 160
        y = 120
        cv2.rectangle(img=frame, 
                        pt1=(x-offset,y-offset), 
                        pt2=(x+width+offset, y+width+offset), 
                        color=(0, 0, 0), 
                        thickness=2
        )     
            
        # get key event
        key = key_action()
            
        if key == 'space':
            # extract the [224x224] rectangle out of it
            image = frame[y:y+width, x:x+width, :]
            predictor(image, model)

        # disable ugly toolbar 
        cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
        # display the resulting frame
        cv2.imshow('frame', frame)            
            
finally:
    # when everything done, release the capture
    logging.info('quit webcam')
    webcam.release()
    cv2.destroyAllWindows()

 