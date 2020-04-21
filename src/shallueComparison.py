from keras.models import load_model
import tensorflow as tf
from lstm import model_creator


def lstm():
    model = load_model('/models/lstm.h5')


def getDataset(filenames):
     dataset = tf.data.TFRecordDataset(filenames,
                                      # or 'GZIP', 'ZLIB' if compress you data.
                                      compression_type=None,
                                      buffer_size=10240,        # any buffer size you want or 0 means no buffering
                                      num_parallel_reads=os.cpu_count()  # or 0 means sequentially reading
                                      )
    dataset = dataset.map(single_example_parser,
                          num_parallel_calls=os.cpu_count())

    dataset = dataset.shuffle(buffer_size=number_larger_than_batch_size)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    return dataset

if __name__ == "__main__":
    filenames_to_read_train = ['tfrecord/train-00000-of-00008', 'tfrecord/train-00001-of-00008', 'tfrecord/train-00002-of-00008', 'tfrecord/train-00003-of-00008',
                         'tfrecord/train-00004-of-00008', 'tfrecord/train-00005-of-00008', 'tfrecord/train-00006-of-00008',
                         'tfrecord/train-00007-of-00008']
    train_dataset = getDataset(filenames_to_read_train)

    # Maybe you want to prefetch some data first.
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Decode the example
    dataset = dataset.map(single_example_parser,
                          num_parallel_calls=os.cpu_count())

    dataset = dataset.shuffle(buffer_size=number_larger_than_batch_size)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
