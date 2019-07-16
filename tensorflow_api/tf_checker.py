import tensorflow as tf
import numpy as np

#for example in tf.python_io.tf_record_iterator("/Users/satrya/Downloads/Custom_Mask_RCNN/training_files/pet_train.record"):

for example in tf.python_io.tf_record_iterator("/Users/satrya/Documents/Github/AI_minor_OP4/data/gestures/tf_records/train-gestures.record-00004-of-00005"):


    result = tf.train.Example.FromString(example)
    #print(result.features) #all features
    #print(result.features.feature['image/encoded'].bytes_list.value)
    print(result.features.feature['image/height'].int64_list.value)
    print(result.features.feature['image/width'].int64_list.value)
    print(result.features.feature['image/filename'].bytes_list.value)
    print(result.features.feature['image/source_id'].bytes_list.value)
    print(result.features.feature['image/format'].bytes_list.value)
    print('xmin: ' + str(result.features.feature['image/object/bbox/xmin'].float_list.value[0] * result.features.feature['image/width'].int64_list.value[0]))
    print('ymax: ' + str(result.features.feature['image/object/bbox/ymax'].float_list.value[0] * result.features.feature['image/height'].int64_list.value[0]))
    print('xmax: ' + str(result.features.feature['image/object/bbox/xmax'].float_list.value[0] * result.features.feature['image/width'].int64_list.value[0]))
    print('ymin: ' + str(result.features.feature['image/object/bbox/ymin'].float_list.value[0] * result.features.feature['image/height'].int64_list.value[0]))
    print(result.features.feature['image/object/difficult'].int64_list.value)
    print(result.features.feature['image/object/truncated'].int64_list.value)
    print(result.features.feature['image/object/class/text'].bytes_list.value)
    print(result.features.feature['image/object/class/label'].int64_list.value)
    print(result.features.feature['image/object/mask'].bytes_list.value)
    print('------------------------------------')
    #test = result.features.feature['image/object/mask'].bytes_list.value[0]


'''for example in tf.python_io.tf_record_iterator("/Users/satrya/Downloads/Custom-Mask-RCNN-using-Tensorfow-Object-detection-API-master/dataset/train.record"):
    result = tf.train.Example.FromString(example)
    print(result.features) #all features

    print(result.features.feature['image/height'].int64_list.value)
    print(result.features.feature['image/width'].int64_list.value)
    print(result.features.feature['image/filename'].bytes_list.value)
    print(result.features.feature['image/source_id'].bytes_list.value)
    print(result.features.feature['image/format'].bytes_list.value)
    print('xmin: ' + str(result.features.feature['image/object/bbox/xmin'].float_list.value))
    print('xmax: ' + str(result.features.feature['image/object/bbox/xmax'].float_list.value))
    print('ymin: ' + str(result.features.feature['image/object/bbox/ymin'].float_list.value))
    print('ymax: ' + str(result.features.feature['image/object/bbox/ymax'].float_list.value))
    print(result.features.feature['image/object/difficult'].int64_list.value)
    print(result.features.feature['image/object/truncated'].int64_list.value)
    print(result.features.feature['image/object/class/text'].bytes_list.value)
    print(result.features.feature['image/object/class/label'].int64_list.value)
    print(result.features.feature['image/object/mask'].bytes_list.value)
    print('------------------------------------')'''
