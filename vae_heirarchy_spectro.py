## source code adapted from Siraj 

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
import datetime
import ipdb
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
#from tensorflow.examples.tutorials.mnist import input_data
#will ensure that the correct data has been downloaded to your
#local training folder and then unpack that data to return a dictionary of DataSet instances.
#mnist = input_data.read_data_sets("MNIST_data/")


BATCH_SIZE = 64
RANDOM_DIM = 100
HEIGHT, WIDTH, CHANNEL, n_pixels = 128, 128, 1, 128 * 128
CHECKPOINT_EVERY = 100   #
EPOCHS = int(1e3)+1   # int(1e5)
LEARNING_RATE = 1e-3
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.1    # 0.3
MAX_TO_KEEP = 20
METADATA = False
CHECK_EVERY = 1
SAVE_EVERY = 50
PLOT_EVERY = 20
slim = tf.contrib.slim    #for testing
gen_eval_num = 20
VERSION =  'FSD_pad'      #newPokemonmnist_pmFSD_pretty_128'newspectrogram' 'OLLO_NO1'
DATA_ROOT = 'training_data'
LOGDIR_ROOT = 'model'
DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
RESULT_DIR_ROOT = 'results'
latent_dim_1 = 100
h_dim_1 = 500  # size of network
latent_dim_2 = 10
h_dim_2 = 100  # size of network


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--version', type=str, default=VERSION,
                        help='The training data group')

    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training steps. Default: ' + str(EPOCHS) + '.')
    #parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        #help='Concatenate and cut audio samples to this many '
                        #'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()

#def open_sess(args):
    #sess = tf.Session()
    #saver = tf.train.Saver(max_to_keep=20)
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    ##ipdb.set_trace()
    ###################### Saver for storing checkpoints of the model.
    #saver = tf.train.Saver(max_to_keep=args.max_checkpoints)
    #try:
        #print(args.restore_from)
        #saved_global_step = load(saver, sess, args.restore_from)
        #if is_overwritten_training or saved_global_step is None:
            ## The first training step will be saved_global_step + 1,
            ## therefore we put -1 here for new or overwritten trainings.
            #saved_global_step = -1
    #except:
        #print("Something went wrong while restoring checkpoint. "
              #"We will terminate training to avoid accidentally overwriting "
              #"the previous model.")
        #raise
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #return saved_global_step

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir))

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

def get_default_logdir(dir_root, version):
    train_dir = os.path.join(dir_root, version, DATESTRING)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print('Using and make default dir: {}'.format(train_dir))
    return train_dir
    
def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = LOGDIR_ROOT
    result_root = RESULT_DIR_ROOT

    version = args.version
    if version is None:
        version = VERSION
        
    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root, version)
        

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir
        print('Restoring from default: {}'.format(restore_from))
        
    result_dir = get_default_logdir(result_root, version)
    print('Saving plots to default: {}'.format(result_dir))
    
    data_dir = os.path.join(DATA_ROOT, version)
    print('Using default data: {}'.format(data_dir))
        
    return {
        'logdir': logdir,
        'restore_from': restore_from,
        'result_dir': result_dir,
        'data_dir': data_dir
    }
        
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 

def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def FC_Layer(X, W, b):
    return tf.matmul(X, W) + b

def plot_test(sess, nodel_no, test_data, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))

    num_pairs = 10
    image_indices = np.random.randint(0, 200, num_pairs)
    #Lets plot 10 digits
    
    x = test_data
    for pair in range(num_pairs):
        #reshaping to show original test image
        x_image = x[pair, :]
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(np.reshape(x_image, (HEIGHT, WIDTH)), cmap="gray_r", aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, WIDTH-1])
        plt.ylim([HEIGHT-1, 0])
        #ipdb.set_trace()
        #reconstructed image, feed the test image to the decoder
        x_image = np.reshape(x[pair, :], (1, -1))
        x_reconstruction = reconstruction.eval(session=sess, feed_dict={X: x_image})
        #reshape it to heightxwidth pixels
        x_reconstruction_image = (np.reshape(x_reconstruction, (HEIGHT, WIDTH)))
        #plot it!
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, cmap="gray_r", aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, WIDTH-1])
        plt.ylim([HEIGHT-1, 0])
        plt.tight_layout()
        if pair == 0 or pair == 1:
            plt.title("Reconstruct")
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")
    plt.close()
    
def process_data():   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, DATA_ROOT, VERSION)
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir, each))

    #  Save the last 16 images for validation
    train_images = tf.convert_to_tensor(images[:-BATCH_SIZE] * EPOCHS, dtype = tf.string)
    test_images = tf.convert_to_tensor(images[-BATCH_SIZE:] * EPOCHS, dtype = tf.string)
    #ipdb.set_trace()
    #valid_images = tf.convert_to_tensor(images[-10:], dtype = tf.string)
    
    #images_queue = tf.train.slice_input_producer(
                                        #[all_images], num_epochs=None)
    #test_images_queue = tf.train.slice_input_producer(
                                        #[test_images], num_epochs=None)                                   
    train_queue = tf.train.string_input_producer(train_images,
                                                        num_epochs=None)
    test_queue = tf.train.string_input_producer(test_images,
                                                        num_epochs=None)
    reader = tf.WholeFileReader()
    train_key, train_content = reader.read(train_queue)
    test_key, test_content = reader.read(test_queue)
    
    train_img = tf.image.decode_png(train_content)
    test_img = tf.image.decode_png(test_content)
    
    #content = tf.read_file(images_queue[0])
    #test_content = tf.read_file(test_images_queue[0])
    #ipdb.set_trace()
    train_image = tf.image.decode_jpeg(train_content, channels = CHANNEL)
    train_image = tf.image.random_brightness(train_image, max_delta = 0.1)
    train_image = tf.image.random_contrast(train_image, lower = 0.9, upper = 1.1)
    
    test_image = tf.image.decode_jpeg(test_content, channels = CHANNEL)
    test_image = tf.image.random_brightness(test_image, max_delta = 0.1)
    test_image = tf.image.random_contrast(test_image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    #size = [1, n_pixels]
    size = [HEIGHT, WIDTH]
    train_image = tf.image.resize_images(train_image, size)
    test_image = tf.image.resize_images(test_image, size)

    train_image = tf.cast(train_image, tf.float32)
    train_image = train_image / 255.0
    test_image = tf.cast(test_image, tf.float32)
    test_image = test_image / 255.0
    
    train_images_batch = tf.train.shuffle_batch(
                                    [train_image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE, min_after_dequeue = 200) #
    test_images_batch = tf.train.shuffle_batch(
                                    [test_image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,min_after_dequeue = 200)     #
    num_images = len(images) * EPOCHS
    #ipdb.set_trace()
    train_images_batch = tf.reshape(train_images_batch, [BATCH_SIZE, n_pixels])
    test_images_batch = tf.reshape(test_images_batch, [BATCH_SIZE, n_pixels])

    return train_images_batch, test_images_batch, num_images

############################ Encoder ############################

"""def encoder_net(x, latent_dim, h_dim):
Construct an inference network parametrizing a Gaussian.
Args:
x: A batch of MNIST digits.
latent_dim: The latent dimensionality.
hidden_size: The size of the neural net hidden layers.
Returns:
mu: Mean parameters for the variational family Normal
sigma: Standard deviation parameters for the variational family Normal
"""
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))
# layer 1
W_enc_1 = weight_variables([n_pixels, h_dim_1], "W_enc_1")
b_enc_1 = bias_variable([h_dim_1], "b_enc_1")
# tanh - activation function        avoid vanishing gradient in generative models
h_enc_1 = tf.nn.tanh(FC_Layer(X, W_enc_1, b_enc_1))

# layer 2   Output mean and std of the latent variable distribution
W_mu_1 = weight_variables([h_dim_1, latent_dim_1], "W_mu_1")
b_mu_1 = bias_variable([latent_dim_1], "b_mu_1")
mu_1 = FC_Layer(h_enc_1, W_mu_1, b_mu_1)

W_logstd_1 = weight_variables([h_dim_1, latent_dim_1], "W_logstd_1")
b_logstd_1 = bias_variable([latent_dim_1], "b_logstd_1")
logstd_1 = FC_Layer(h_enc_1, W_logstd_1, b_logstd_1)

# Reparameterize import Randomness
noise = tf.random_normal([1, latent_dim_1])
# z_1 is the fisrt leverl output(latent variable) of our Encoder
z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*logstd_1))

# second level---------------------------- layer 1
W_enc_2 = weight_variables([latent_dim_1, h_dim_2], "W_enc_2")
b_enc_2 = bias_variable([h_dim_2], "b_enc_2")
# tanh - activation function        avoid vanishing gradient in generative models
h_enc_2 = tf.nn.tanh(FC_Layer(z_1, W_enc_2, b_enc_2))

# layer 2   Output mean and std of the latent variable distribution
W_mu_2 = weight_variables([h_dim_2, latent_dim_2], "W_mu_2")
b_mu_2 = bias_variable([latent_dim_2], "b_mu_2")
mu_2 = FC_Layer(h_enc_2, W_mu_2, b_mu_2)

W_logstd_2 = weight_variables([h_dim_2, latent_dim_2], "W_logstd_2")
b_logstd_2 = bias_variable([latent_dim_2], "b_logstd_2")
logstd_2 = FC_Layer(h_enc_2, W_logstd_2, b_logstd_2)

# Reparameterize import Randomness
noise_2 = tf.random_normal([1, latent_dim_2])
# z_1 is the ultimate output(latent variable) of our Encoder
z_2 = mu_2 + tf.multiply(noise_2, tf.exp(0.5*logstd_2))

############################ Dencoder ############################
"""Build a generative network parametrizing the likelihood of the data
Args:
z: Samples of latent variables
hidden_size: Size of the hidden state of the neural net
Returns:
bernoulli_logits: logits for the Bernoulli likelihood of the data
"""
# layer 1
W_dec_2 = weight_variables([latent_dim_2, h_dim_2], "W_dec_2")
b_dec_2 = bias_variable([h_dim_2], "b_dec_2")
# tanh - decode the latent representation
h_dec_2 = tf.nn.tanh(FC_Layer(z_2, W_dec_2, b_dec_2))

# layer2 - reconstruction the first leverl latent variables
W_rec_2 = weight_variables([h_dim_2, latent_dim_1], "W_dec_2")
b_rec_2 = bias_variable([latent_dim_1], "b_rec_2")
recon_z1 = tf.nn.sigmoid(FC_Layer(h_dec_2, W_rec_2, b_rec_2)) # ?????

# layer 1
W_dec_1 = weight_variables([latent_dim_1, h_dim_1], "W_dec")
b_dec_1 = bias_variable([h_dim_1], "b_dec")
# tanh - decode the latent representation

#ipdb.set_trace()
residual_z1 = tf.identity(z_1) + recon_z1
h_dec_1 = tf.nn.tanh(FC_Layer(residual_z1 , W_dec_1, b_dec_1))

# layer2 - reconstruction the image and output 0 or 1
W_rec_1 = weight_variables([h_dim_1, n_pixels], "W_rec_1")
b_rec_1 = bias_variable([n_pixels], "b_rec_1")
# 784 bernoulli parameter Output
reconstruction = tf.nn.sigmoid(FC_Layer(h_dec_1, W_rec_1, b_rec_1))

def train():
    random_dim = RANDOM_DIM

    ################### Get parameters
    args = get_arguments()
    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']
    result_dir = directories['result_dir']
    data_dir = directories['data_dir']
    #x_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='x_placeholder')
    
    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    ################### define graph
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, n_pixels, CHANNEL], name='real_image')
        #real_image = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # #### Loss
    log_likelihood1 = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9))
    
    KL_divergence1 = -0.5 * tf.reduce_sum(1 + 2*logstd_2 - tf.pow(mu_2, 2) - tf.exp(2 * logstd_2), reduction_indices=1)
    
    VAE_loss = tf.reduce_mean(log_likelihood1 - KL_divergence1)

    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_divergence1', KL_divergence1)
    tf.summary.scalar('log_likelihood1', log_likelihood1)

    # ### Optimizer
    optimizer = tf.train.AdadeltaOptimizer(0.0005).minimize(- VAE_loss)   # 
    
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    ################################### load data
    batch_size = BATCH_SIZE
    image_batch, test_batch, samples_num = process_data()   # ?????????????
    batch_num = int(samples_num / batch_size)
    
    ##################### Set up session and check the glabal save steps
    #saved_global_step = open_sess(args)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #ipdb.set_trace()
    ##################### Saver for storing checkpoints of the model.
    saver = tf.train.Saver(max_to_keep=args.max_checkpoints)
    try:
        print(args.restore_from)
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #################### TRaining
    print('start training...')
    last_saved_step = saved_global_step
    step = None
    print("last_saved_step: ", last_saved_step)
    variational_lower_bound_array = []
    log_likelihood_array1 = []
    log_likelihood_array2 = []
    KL_term_array1 = []
    KL_term_array2 = []
    iteration_array = [i*CHECK_EVERY for i in range(EPOCHS/CHECK_EVERY)]
    for ii in range(saved_global_step + 1, EPOCHS):
        t1 = time.time()
        save_name = result_dir + '/' + "VAE_FSD_PAD_step{}_".format(ii)
        print("epoch: ", ii)
        # load training batch
        #ipdb.set_trace()
        for batch in range(batch_num - 1):
            if batch == np.int(batch_num/2):
                print "half way in one epoch"
            x_batch = sess.run(image_batch)
            t_batch = sess.run(test_batch)
            
            #run our optimizer on our data
            sess.run(optimizer, feed_dict={X: x_batch})

        if (ii % CHECK_EVERY == 0):
            #every 1K iterations record these values
            vlb_eval = VAE_loss.eval(session=sess, feed_dict={X: x_batch})
            print "Iteration: {}, Loss: {}".format(ii, vlb_eval)
            variational_lower_bound_array.append(vlb_eval)
            log_likelihood_array1.append(np.mean(log_likelihood1.eval(session=sess, feed_dict={X: x_batch})))
            KL_term_array1.append(np.mean(KL_divergence1.eval(session=sess, feed_dict={X: x_batch})))
            #log_likelihood_array2.append(np.mean(log_likelihood1.eval(feed_dict={X: x_batch})))
            #KL_term_array2.append(np.mean(KL_divergence2.eval(feed_dict={X: x_batch})))
        t1 = time.time()
            
        if (ii % PLOT_EVERY== 0):
            #plot_prior(ii)
            #ipdb.set_trace()
            plot_test(sess, ii, t_batch, save_name=save_name)

            #plot error
            iteration_array = [i*CHECK_EVERY for i in range(ii+1)]
            plt.figure()
            np.savez("losses_file.npz", )
            plt.plot(iteration_array, variational_lower_bound_array, "b*-")
            plt.plot(iteration_array, KL_term_array1, "co-")
            plt.plot(iteration_array, log_likelihood_array1, "md-")
            plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], loc="best")
            plt.title('Loss per iteration')
            plt.savefig(save_name+"loss.png", format="png")
            plt.close()
    
            # plot posterior predictive space
            # Get fixed MNIST digits for plotting posterior means during training
            #np_x_fixed = t_batch
            #np_x_fixed = np_x_fixed.reshape(16, n_pixels)
            #np_x_fixed = (np_x_fixed > 0.5).astype(np.float32)
            #np_q_mu = sess.run(mu_1, {X: np_x_fixed})
            #cmap = "gray_r"
            #f, ax = plt.subplots(1, figsize=(6 * 1.1618, 6))
            #im = ax.scatter(np_q_mu[:, 0], np_q_mu[:, 1], cmap=cmap, alpha=0.7)
            #ax.set_xlabel('First dimension of sampled latent variable $z_1$')
            #ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
            ##ax.set_xlim([-10., 10.])
            ##ax.set_ylim([-10., 10.])
            #f.colorbar(im, ax=ax, label='Digit class')
            #plt.tight_layout()
            #plt.savefig(save_name + '_posterior_predictive_map_frame_{}.png'.format(ii), format="png")
            #plt.close()
            
            #nx = ny = 20
            #x_values = np.linspace(-3, 3, nx)
            #y_values = np.linspace(-3, 3, ny)
            #canvas = np.empty((height * ny, width * nx))
            #for ii, yi in enumerate(x_values):
              #for j, xi in enumerate(y_values):
                #np_z = np.expand_dims(np.append(np.ones((8)), np.array([[xi, yi]])), axis=0)
                ##ipdb.set_trace()
                #x_mean = sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
                #canvas[(nx - ii - 1) * height:(nx - ii) * height,  j *
                       #width:(j + 1) * width] = x_mean[0].reshape(height, width)
            #plt.savefig(save_name + '_prior_predictive_map_frame_{}'.format(ii), format="png")   # canvas
            #plt.close()

        # save check point every 500 epoch
        if i % SAVE_EVERY == 0:
            save(saver, sess, logdir, step)
            last_saved_step = step
        #sess.reset(process_data, [image_batch, test_batch, samples_num])        
        coord.request_stop()
        coord.join(threads)

    



if __name__ == "__main__":

    train()

