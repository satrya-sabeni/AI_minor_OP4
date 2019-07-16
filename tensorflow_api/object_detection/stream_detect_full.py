from utils import label_map_util
from utils import visualization_utils as vis_util
import numpy as np
import pyautogui
import tensorflow as tf

import cv2
cap = cv2.VideoCapture(0)

MODEL = './trained_model/gestures/frozen_inference_graph.pb'
LABELS = './trained_model/gestures/label_map.pbtxt'
NUM_CLASSES = 1

pyautogui.moveTo(1920/2,1080/2)

# Load Tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            pyautogui.FAILSAFE = False
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                categories_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.7,
                line_thickness=8)

            '''for c, score in zip(classes[0], scores[0]):
                category = categories[int(c) - 1]['name']
                if(score > 0.7 and category == 'openhand'):
                    print(int(c), ':', score)
              
                    print(boxes[0][0])
                    x = ((boxes[0][0][1]-boxes[0][0][0])/2) + boxes[0][0][0]
                    y = ((boxes[0][0][2]-boxes[0][0][3])/2) + boxes[0][0][3]
                    pyautogui.moveTo(x*1920,y*1080)
                    # print(boxes[0][0][1] * 800)
                    # print(boxes[0][0][0] * 600)'''


            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # main()
            