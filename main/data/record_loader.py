import tensorflow as tf
import io
import numpy as np

train_rec = '/Volumes/MyPassport/planes_scannet_train.tfrecords'
val_rec = '/Volumes/MyPassport/planes_scannet_val.tfrecords'

HEIGHT = 192
WIDTH = 256
NUM_PLANES = 20

NUM_TRAIN = 50000
NUM_VAL = 760

def extract_image(feature):
    img_string = feature['image_raw'].bytes_list.value[0]
    buf = io.BytesIO(img_string)
    np_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.reshape(np_arr, (1, HEIGHT, WIDTH, 3))
    return img


def extract_seg(feature):
    img_string = feature['segmentation_raw'].bytes_list.value[0]
    buf = io.BytesIO(img_string)
    np_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.reshape(np_arr, (1, HEIGHT, WIDTH))
    return img


def extract_semantics(feature):
    img_string = feature['segmentation_raw'].bytes_list.value[0]
    buf = io.BytesIO(img_string)
    np_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.reshape(np_arr, (1, HEIGHT, WIDTH))
    return img


def extract_num_planes(feature):
    num = np.array(feature['num_planes'].int64_list.value[0])
    return np.reshape(num, (1, 1))


def extract_planes(feature):
    planes = np.array(feature['plane'].float_list.value)
    return np.reshape(planes, (1, NUM_PLANES, 3))


def extract_depth(feature):
    planes = np.array(feature['depth'].float_list.value)
    return np.reshape(planes, (1, HEIGHT, WIDTH))


class RecordLoader:

    def __init__(self, rec_path, rec_type, batch_size, num_workers=4):
        self.rec_path = rec_path
        self.record_iterator = tf.python_io.tf_record_iterator(path=rec_path)
        self.length = NUM_TRAIN if rec_type == 'train' else NUM_VAL
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        self.record_iterator = tf.python_io.tf_record_iterator(path=self.rec_path)
        return self

    def __len__(self):
        return self.length

    def __next__(self):

        # placehold_args = [None for i in range(self.batch_size)]
        # with Pool(self.num_workers) as p:
        #     data = p.map(self.load_next_features, placehold_args)

        data = []
        for i in range(self.batch_size):
            frame_data = self.load_next_features()
            data.append(frame_data)

        return self.assemble_batch(data)

    def assemble_batch(self, data):
        batch = {}
        feature_keys = data[0].keys()
        for key in feature_keys:
            batch_data = np.concatenate([frame_data[key] for frame_data in data])
            batch[key] = batch_data
        return batch

    def load_next_features(self):
        string_record = next(self.record_iterator)
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feature = example.features.feature
        return {
            'image_raw': extract_image(feature),
            'segmentation_raw': extract_seg(feature),
            'num_planes': extract_num_planes(feature),
            'plane': extract_planes(feature),
            'depth': extract_depth(feature)
        }
