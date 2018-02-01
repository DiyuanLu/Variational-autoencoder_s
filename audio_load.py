########### script for load data in tensorflow
# reference https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/


import tensorflow as tf
import ipdb
import os
import codecs
import csv
import re
from audio_hyperparams import Hyperparams as hp
import numpy as np
from functools import wraps
import threading
from tensorflow.python.platform import tf_logging as logging
import copy
import librosa
import audio_utils as utils
import fnmatch

#### prepare_data
def find_files(directory, pattern='*.wav'):
    '''Recursively finds ALL files matching the pattern in the given directory.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

#### find_files is a better replacement for reading csv
def load_sound_fnames():
    '''put audio file names in a .csv file'''
    audio_fpath = '/home/elu/LU/2_Neural_Network/2_NN_projects_codes/tacotron/WEB/'
    sound_files = []
    reader = csv.reader(codecs.open(audio_fpath + "audio_files.csv", 'rb', 'utf-8'))
    for ind, row in enumerate(reader):
        sound_fname, _ = row
        sound_file = hp.sound_fpath + "/" + sound_fname  # + ".wav" 
        sound_files.append(sound_file)
        #if hp.min_len <= duration <= hp.max_len:
            ## this returns the 16 jinzhi number each in 4 bits
            #sound_files.append(sound_queue)
            #durations.append(duration)
    return sound_files
     
def load_train_data(is_training=True):
    """We train on the whole data but the last num_samples."""
    
    sound_files = find_files(hp.sound_fpath)
    if is_training:
        if hp.sanity_check: # We use a single mini-batch for training to overfit it.
            sound_files = sound_files[:hp.batch_size]*1000
        else:
            sound_files = sound_files[:-hp.batch_size]
    else:
        if hp.sanity_check: # We use a single mini-batch for training to overfit it.
            sound_files = sound_files[:hp.batch_size]
        else:
            sound_files = sound_files[-hp.batch_size:]
            
    return sound_files

################################ load data
# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.

    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1


def get_batch(is_training=True):
    """Loads training data and put them in queues
    return;
        spectro: (batch_size, length, n_mels)
        magnit: (batch_size, length, 5*)"""
    with tf.device('/cpu:0'):
        # Load data   texts, sound_files = texts[:-batch_size], sound_files[:-batch_size]
        sound_files = load_train_data(is_training=is_training) # byte, string

        # calc total batch count
        num_batch = len(sound_files) // hp.batch_size
         
        # Convert to tensor
        sound_files_tensor = tf.convert_to_tensor(sound_files, dtype = tf.string)     # .wav files
         
        # Create Queues
        '''TypeError: 'Tensor' object is not iterable. need to put[] around the tensor'''
        sound_queue, sound_queue2 = tf.train.slice_input_producer([sound_files_tensor, sound_files_tensor], shuffle=True)   # syntax with [], otherwise "tensor object" is not iteratable
        ##### TODO there must be a way to solve the problem!
 
        ##### Optional data processing 
        @producer_func
        def get_audio_spectrograms(_inputs):
            ''' Dealing with sentence by sentence
            From `_inputs`, which has been fetched from slice queues,
               makes text, spectrogram, and magnitude,
               then enqueue them again. 
            '''
            _sound_file, _ = _inputs   # sentence and .wav file
            
            # Processing
            _spectrogram, _magnitude, _length = utils.get_spectrograms(_sound_file)
            
            _spectrogram = utils.reduce_frames(_spectrogram, hp.win_length//hp.hop_length, hp.r)
            _magnitude = utils.reduce_frames(_magnitude, hp.win_length//hp.hop_length, hp.r)
    
            return _spectrogram, _magnitude, _length
            
        spec, magnit, length = get_audio_spectrograms(inputs=[sound_queue, sound_queue2], 
                                            dtypes=[tf.float32, tf.float32, tf.int32],
                                            capacity=128,
                                            num_threads=2)
        
        # create batch queues  An input tensor with shape [x, spec, magnit] will be output as a tensor with shape [batch_size, x, spec, magnit]
        spec, magnit, length = tf.train.batch([spec, magnit, length],
                                shapes=[(None, hp.n_mels*hp.r), (None, (1+hp.n_fft//2)*hp.r), (None,)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=True)
        
        if hp.use_log_magnitude:
            magnit = tf.log(magnit+1e-10)
            
    return spec, magnit, length, num_batch
    #return sound_batch
 



#def main():
    #ipdb.set_trace()
    #spec, magnit, length, num_batch = get_batch()
    ##sound_batch = get_batch()
    #coord = tf.train.Coordinator()
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())

    ##coord = tf.train.Coordinator() # OMG!!! This line should NOT be here!
    #threads = tf.train.start_queue_runners(sess=sess)

    #for k in range(10):
        #spectro, magnit, length = sess.run([spec, magnit, length])



    
#if __name__ == "__main__":
    #load_train_data()







