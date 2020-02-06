import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from object_detection.utils import ops as utils_ops
from PIL import Image
import os
import person_detector


IMAGE_FOLDER = "./images/unclassified"
POS_FOLDER = "./images/classified/positive"
NEG_FOLDER = "./images/classified/negative"



if __name__ == "__main__":
    detection_graph = person_detector.open_graph()

    images = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]
    positive_images = filter(lambda image: (image.startswith("1_")), images)
    negative_images = filter(lambda image: (image.startswith("0_")), images)

    with detection_graph.as_default():
        with tf.Session() as sess:

            for pos in positive_images:
                old_filename = IMAGE_FOLDER + "/" + pos
                new_filename = POS_FOLDER + "/" + pos[:-5] + ".jpg"
                if not os.path.isfile(new_filename):
                    img = person_detector.get_person(old_filename, sess)
                    if not img:
                        continue
                    img = img.convert('L')
                    img.save(new_filename, "jpeg")

            for neg in negative_images:
                old_filename = IMAGE_FOLDER + "/" + neg
                new_filename = NEG_FOLDER + "/" + neg[:-5] + ".jpg"
                if not os.path.isfile(new_filename):
                    img = person_detector.get_person(old_filename, sess)
                    if not img:
                        continue
                    img = img.convert('L')
                    img.save(new_filename, "jpeg")

