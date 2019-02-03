import tensorflow as tf
import functools

from c3d.classification.dataset import ImageDataset


class TFImageDataset:
    def __init__(self, image_dataset: ImageDataset, batch_size, n_epochs=None,
                 shuffle_size=100000, max_workers=10, batch_prefetch=1,
                 img_shape=None):
        self.image_dataset = image_dataset
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_workers = max_workers
        self.batch_prefetch = batch_prefetch
        self.img_shape = img_shape

        self._paths_placeholder = None
        self._labels_placeholder = None

        self._train_dataset = None
        self._train_iter = None
        self._test_dataset = None
        self._test_iter = None

        self._data_iter = None

        self.dataspec_placeholder = None
        self.train_spec = None
        self.test_spec = None

        self.batch_images = None
        self.batch_labels = None

    @staticmethod
    def _read_img(path, shape=None):
        im_str = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(im_str, channels=3)
        assert image_decoded.dtype == tf.uint8
        if shape is not None:
            image_decoded = tf.reshape(image_decoded, shape)
        return image_decoded / 255

    @staticmethod
    def _color_augment(img):
        # Luminance shift
        lum_mult = tf.random_uniform(
            minval=0.8, maxval=1.2, shape=(1, 1, 1)
        )
        # Color balance shift
        color_mult = tf.random_uniform(
            minval=0.9, maxval=1.1, shape=(1, 1, 3)
        )
        # Recolor by both shifts
        return tf.minimum(
            1.,
            img * (lum_mult * color_mult)
        )

    @staticmethod
    def _random_flip(img):
        prob = tf.random_uniform((), minval=0, maxval=1)
        return tf.cond(
            prob > 0.5,
            lambda: tf.image.flip_left_right(img),
            lambda: img
        )

    @classmethod
    def _prepare_image(cls, path, is_training: bool, shape=None):
        # THIS METHOD IS INTENTIONALLY NOT STATIC, TO ENABLE OVERRIDE!
        img = cls._read_img(path, shape=shape)
        if not is_training:
            img = cls._color_augment(img)
            img = cls._random_flip(img)
        return img

    @classmethod
    def _prepare_data(cls, path, label, is_training: bool, shape=None):
        return cls._prepare_image(path, is_training, shape=shape), label

    def prepare_single_dataset(self, dataset, is_training: bool):
        return dataset

    def prepare_batch_dataset(self, dataset, is_training: bool):
        return dataset

    def _make_dataset(self, is_training: bool, repeat=False, n_epochs=None):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._paths_placeholder, self._labels_placeholder)
        )
        dataset = dataset.shuffle(self.shuffle_size)
        if repeat:
            dataset = dataset.repeat(count=n_epochs)
        dataset = dataset.map(functools.partial(self._prepare_data,
                                                is_training=is_training,
                                                shape=self.img_shape),
                              num_parallel_calls=self.max_workers)
        dataset = self.prepare_single_dataset(dataset, is_training=is_training)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = self.prepare_batch_dataset(dataset, is_training=is_training)
        dataset = dataset.prefetch(self.batch_prefetch)
        return dataset

    def build(self):
        self._paths_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
        self._labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

        self._train_dataset = self._make_dataset(is_training=True,
                                                 n_epochs=self.n_epochs, repeat=True)
        self._train_iter = self._train_dataset.make_initializable_iterator()

        self._test_dataset = self._make_dataset(is_training=False, repeat=False)
        self._test_iter = self._test_dataset.make_initializable_iterator()

        self.dataspec_placeholder = tf.placeholder(tf.string, shape=[])
        self._data_iter = tf.data.Iterator.from_string_handle(
            self.dataspec_placeholder,
            self._train_dataset.output_types, self._train_dataset.output_shapes
        )

        data_item = self._data_iter.get_next()
        self.batch_images, self.batch_labels = data_item[0], data_item[1]

    def _get_init_args(self, file_ids, label2index):
        file_ids = list(file_ids)
        return {
            self._paths_placeholder: [self.image_dataset.file_path(f_id)
                                      for f_id in file_ids],
            self._labels_placeholder: [label2index[f_id.label]
                                       for f_id in file_ids],

        }

    def initialize_train(self, tf_session: tf.Session, label2index):
        tf_session.run(
            self._train_iter.initializer, self._get_init_args(
                self.image_dataset.file_ids_train_iter(), label2index
            )
        )
        self.train_spec = tf_session.run(self._train_iter.string_handle())

    def initialize_test(self, tf_session: tf.Session, label2index):
        tf_session.run(
            self._test_iter.initializer, self._get_init_args(
                self.image_dataset.file_ids_test_iter(), label2index
            )
        )
        self.test_spec = tf_session.run(self._test_iter.string_handle())
