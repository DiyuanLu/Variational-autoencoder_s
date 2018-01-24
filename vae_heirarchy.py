# source code adapted from Siraj 

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time #lets clock training time..
import datetime
import os
import argparse
#from imageio import imwrite
#import data
from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets    #read data
import ipdb
mnist = input_data.read_data_sets("training_data/MNIST_data/")

# HyperParameters
BATCH_SIZE = 64
latent_dim_1 = 200
latent_dim_2 = 50
h_dim_1 = 1000
h_dim_2 = 300
HEIGHT, WIDTH, CHANNEL = 28, 28, 1
n_pixels = HEIGHT * WIDTH
EPOCHS = int(1e4)   # int(1e5)
LEARNING_RATE = 1e-3
SILENCE_THRESHOLD = 0.1    # 0.3
MAX_TO_KEEP = 20
METADATA = False
CHECKPOINT_EVERY = 500   #
CHECK_EVERY = 10
SAVE_EVERY = 1000
PLOT_EVERY = 5

gen_eval_num = 20

VERSION =  "MNIST_2level-end2endLoss"
DATA_DIR = 'training_data/MNIST_data'
LOGDIR_ROOT = 'model'
DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
RESULT_DIR_ROOT = 'results'

end2end = True

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--version', type=str, default=VERSION,
                        help='The network structure')
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
    
    data_dir = DATA_DIR
    print('Using default data: {}'.format(data_dir))
        
    return {
        'logdir': logdir,
        'restore_from': restore_from,
        'result_dir': result_dir,
        'data_dir': data_dir
    }

    
def process_data():   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'image_data', VERSION)
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images
    #  Save the last 10 images for validation 
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    #valid_images = tf.convert_to_tensor(images[-10:], dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()
    
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    images_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return images_batch, num_images
    
def highwaynet(inputs, h_dim=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      h_dim: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not h_dim:
        h_dim = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=h_dim, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=h_dim, activation=tf.nn.sigmoid, name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C
    return outputs


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def FC_Layer(X, W, b):
    return tf.matmul(X, W) + b

def plot_test(sess, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))

    num_pairs = 10
    image_indices = np.random.randint(0, 200, num_pairs)
    #Lets plot 10 digits
    
    for pair in range(num_pairs):
        #reshaping to show original test image
        x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
        x_image = np.reshape(x, (HEIGHT, WIDTH))
        
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(x_image, aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, WIDTH-1])
        plt.ylim([HEIGHT-1, 0])
        ipdb.set_trace()
        #reconstructed image, feed the test image to the decoder
        #x_reconstruction = reconstruction.eval(feed_dict={real_data: x})
        x_reconstruction = sess.run(reconstruction, feed_dict={real_data: x})
        #reshape it to 28x28 pixels
        x_reconstruction_image = (np.reshape(x_reconstruction, (height, width)))
        #plot it!
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, width-1])
        plt.ylim([height-1, 0])
        plt.tight_layout()
        if pair == 0 or pair == 1:
            plt.title("Reconstruct")
    #ipdb.set_trace()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")
    plt.close()


def plot_prior(model_No, load_model=False):
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    nx = ny = 5     
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((height * ny, width * nx))
    noise = tf.random_normal([1, latent_dim_2])
    z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    
    #ipdb.set_trace()
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        z[0:2] = np.array([[xi, yi]])  #sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
        x_reconstruction = reconstruction.eval(feed_dict={z: z})
        ## layer 1
        #W_dec = weight_variables([latent_dim, h_dim], "W_dec")
        #b_dec = bias_variable([h_dim], "b_dec")
        ## tanh - decode the latent representation
        #h_dec = tf.nn.tanh(FC_Layer(z, W_dec, b_dec))

        ## layer2 - reconstruction the image and output 0 or 1
        #W_rec = weight_variables([h_dim, n_pixels], "W_dec")
        #b_rec = bias_variable([n_pixels], "b_rec")
        ## 784 bernoulli parameter Output
        #reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
    
        canvas[(nx - ii - 1) * height:(nx - ii) * height, j *
               width:(j + 1) * width] = reconstruction[0].reshape(height, width)
    plt.savefig(os.path.join(logdir,
                        'prior_predictive_map_frame_%d.png' % model_No), format="jpg")   # canvas
                        
############################ Encoder ############################
def encoder(real_data):
    """def encoder_net(x, latent_dim, h_dim):
    Construct an inference network parametrizing a Gaussian.
    Args:
    x: A batch of real data (MNIST digits).
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
    Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
    """
    h_dim_1, latent_dim_1, h_dim_2, latent_dim_2 = 1000, 200, 300, 50 #
    
    with tf.variable_scope('enc') as scope:
        # layer 1
        W_enc_1 = weight_variables([n_pixels, h_dim_1], "W_enc_1")
        b_enc_1 = bias_variable([h_dim_1], "b_enc_1")
        # tanh - activation function        avoid vanishing gradient in generative models
        h_enc_1 = tf.nn.tanh(FC_Layer(real_data, W_enc_1, b_enc_1))

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

        return mu_1, logstd_1, mu_2, logstd_2, z_1

############################ Dencoder ############################
def decoder(random_input, z_1):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    z: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
    """
    h_dim_1, latent_dim_1, h_dim_2, latent_dim_2 = 1000, 200, 300, 50 # channel num

    with tf.variable_scope('dec') as scope:
        # layer 1
        W_dec_2 = weight_variables([latent_dim_2, h_dim_2], "W_dec_2")
        b_dec_2 = bias_variable([h_dim_2], "b_dec_2")
        # tanh - decode the latent representation
        h_dec_2 = tf.nn.tanh(FC_Layer(random_input, W_dec_2, b_dec_2))

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

        return reconstruction

def train():
    random_dim = latent_dim_2
    
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
    
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_data = tf.placeholder(tf.float32, shape = [None, n_pixels], name='real_image')
        #real_image = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
        random_input = tf.placeholder(tf.float32, shape=[None, latent_dim_2], name='rand_input')

    # graph
    mu_1, sigma_1, mu_2, sigma_2, z_1 = encoder(real_data)
    reconstruction = decoder(random_input, z_1)

    # #### Loss
    if end2end:
        log_likelihood1 = tf.reduce_sum(real_data * tf.log(reconstruction + 1e-9) + (1 - real_data) * tf.log(1 - reconstruction + 1e-9))

        KL_divergence1 = -0.5 * tf.reduce_sum(1 + 2*sigma_2 - tf.pow(mu_2, 2) - tf.exp(2 * sigma_2), reduction_indices=1)

        VAE_loss = tf.reduce_mean(log_likelihood1 - KL_divergence1)
        #Outputs a Summary protocol buffer containing a single scalar value.
        tf.summary.scalar('VAE_loss', VAE_loss)
        tf.summary.scalar('log_likelihood', log_likelihood1)
        tf.summary.scalar('KL_divergence', KL_divergence1)
    
    else:
        log_likelihood1 = tf.reduce_sum(real_data * tf.log(reconstruction + 1e-9) + (1 - real_data) * tf.log(1 - reconstruction + 1e-9))

        KL_divergence1 = -0.5 * tf.reduce_sum(1 + 2*logstd_1 - tf.pow(mu_1, 2) - tf.exp(2 * logstd_1), reduction_indices=1)

        # latent z1 and reconstruced latent z1
        log_likelihood2 = tf.reduce_sum(z_1 * tf.log(recon_z1 + 1e-9) + (1 - z_1) * tf.log(1 - recon_z1 + 1e-9))

        KL_divergence2 = -0.5 * tf.reduce_sum(1 + 2*logstd_2 - tf.pow(mu_2, 2) - tf.exp(2 * logstd_2), reduction_indices=1)

        VAE_loss1 = tf.reduce_mean(log_likelihood1 - KL_divergence1)
        VAE_loss2 = tf.reduce_mean(log_likelihood2 - KL_divergence2)
        VAE_loss = VAE_loss1 + VAE_loss2

        #Outputs a Summary protocol buffer containing a single scalar value.
        tf.summary.scalar('VAE_loss', VAE_loss)
        tf.summary.scalar('VAE_loss1', VAE_loss1)
        tf.summary.scalar('VAE_loss2', VAE_loss2)
        tf.summary.scalar('log_likelihood1', log_likelihood1)
        tf.summary.scalar('log_likelihood2', log_likelihood2)
        tf.summary.scalar('KL_divergence1', KL_divergence1)
        tf.summary.scalar('KL_divergence2', KL_divergence2)
    
    # ### Optimizer
    #t_vars = tf.trainable_variables()
    #enc_vars = [var for var in t_vars if 'enc' in var.name]
    #dec_vars = [var for var in t_vars if 'dec' in var.name]
    optimizer = tf.train.AdadeltaOptimizer(0.0005).minimize(- VAE_loss) 

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    ############### load data
    #batch_size = BATCH_SIZE
    #image_batch, samples_num = process_data()

    #batch_num = int(samples_num / batch_size)
    #total_batch = 0

    # load data
    mnist = read_data_sets(data_dir, one_hot=True)

    # Set up session
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(max_to_keep=args.max_checkpoints)
    try:
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

    #store value for these 3 terms so we can plot them later
    variational_lower_bound_array = []
    log_likelihood_array1 = []
    log_likelihood_array2 = []
    KL_term_array1 = []
    KL_term_array2 = []
    iteration_array = [i*CHECK_EVERY for i in range(EPOCHS/CHECK_EVERY)]

    print('start training...')
    step = None
    last_saved_step = saved_global_step
    print("last_saved_step: ", last_saved_step)
    #ipdb.set_trace()
    with sess.as_default():
        for ii in range(saved_global_step + 1, EPOCHS):
            save_name = result_dir + '/' + "step{}_".format(ii)
            train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
            #x_batch = sess.run(image_batch)
            x_batch = np.round(mnist.train.next_batch(BATCH_SIZE)[0])
            #run our optimizer on our data
            #ipdb.set_trace()
            sess.run(optimizer, feed_dict={real_data: x_batch, random_input: train_noise})
            #ipdb.set_trace()
            if (ii % 10 == 0):
                #every 1K iterations record these values
                vlb_eval = VAE_loss.eval(session=sess, feed_dict={real_data: x_batch, random_input: train_noise})
                print "Iteration: {}, Loss: {}".format(ii, vlb_eval)
                variational_lower_bound_array.append(vlb_eval)
                log_likelihood_array1.append(np.mean(log_likelihood1.eval(session=sess,feed_dict={real_data: x_batch, random_input: train_noise})))
                KL_term_array1.append(np.mean(KL_divergence1.eval(session=sess,feed_dict={real_data: x_batch, random_input: train_noise})))
                #log_likelihood_array2.append(np.mean(log_likelihood1.eval(feed_dict={real_data: x_batch})))
                #KL_term_array2.append(np.mean(KL_divergence2.eval(feed_dict={real_data: x_batch})))
            t1 = time.time()
            
            if (ii % SAVE_EVERY == 0):
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                saver.save(sess, logdir + '/' + str(ii))
                
            if (ii % PLOT_EVERY == 0):
                #plot_prior(ii)
                plot_test(sess, save_name=save_name)

                # plot posterior predictive space
                # Get fixed MNIST digits for plotting posterior means during training
                np_x_fixed, np_y = mnist.test.next_batch(100)
                np_x_fixed = np_x_fixed.reshape(100, n_pixels)
                np_x_fixed = (np_x_fixed > 0.5).astype(np.float32)
                np_q_mu = sess.run(mu_1, {real_data: np_x_fixed})
                cmap = "jet"
                f, ax = plt.subplots(1, figsize=(6 * 1.1618, 6))
                im = ax.scatter(np_q_mu[:, 0], np_q_mu[:, 1], c=np.argmax(np_y, 1), cmap=cmap, alpha=0.7)
                ax.set_xlabel('First dimension of sampled latent variable $z_1$')
                ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
                ax.set_xlim([-10., 10.])
                ax.set_ylim([-10., 10.])
                f.colorbar(im, ax=ax, label='Digit class')
                plt.tight_layout()
                plt.savefig(save_name + '_posterior_predictive_map_frame_{}.png'.format(ii), format="png")
                plt.close()
                
                nx = ny = 20
                x_values = np.linspace(-3, 3, nx)
                y_values = np.linspace(-3, 3, ny)
                canvas = np.empty((height * ny, width * nx))
                for ii, yi in enumerate(x_values):
                  for j, xi in enumerate(y_values):
                    np_z = np.expand_dims(np.append(np.ones((8)), np.array([[xi, yi]])), axis=0)
                    #ipdb.set_trace()
                    x_mean = sess.run(reconstruction, {random_input: np_z, real_data: np_x_fixed})
                    canvas[(nx - ii - 1) * height:(nx - ii) * height, j *
                           width:(j + 1) * width] = x_mean[0].reshape(height, width)
                plt.savefig(save_name + '_prior_predictive_map_frame_{}'.format(ii), format="png")   # canvas
                plt.close()

        plt.figure()
        #for the number of iterations we had 
        #plot these 3 terms
        np.savez("losses_file.npz", )
        plt.plot(iteration_array, variational_lower_bound_array)
        plt.plot(iteration_array, KL_term_array)
        plt.plot(iteration_array, log_likelihood_array)
        plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
        plt.title('Loss per iteration')
        plt.savefig(save_name+"loss.png", format="png")
        plt.close()

if __name__ == "__main__":

    train()
