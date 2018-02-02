## source code adapted from Siraj 

import os
import sys
import tensorflow as tf
import numpy as np
#import cv2
import random
import scipy.misc
from utils import *

import ipdb
import time
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from audio_load import get_batch
from audio_hyperparams import Hyperparams as hp


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--version', type=str, default=hp.VERSION,
                        help='The training data group')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
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
    train_dir = os.path.join(dir_root, version, hp.DATESTRING)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print('Using and make default dir: {}'.format(train_dir))
    return train_dir
    
def validate_directories(args):
    """Validate and arrange directory related arguments."""

    ## Validation
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
    logdir_root = hp.LOGDIR_ROOT
    result_root = hp.RESULT_DIR_ROOT

    version = args.version
    if version is None:
        version = hp.VERSION
        
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
    
    data_dir = os.path.join(hp.DATA_ROOT, version)
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

#def FC_Layer(X, W, b):
    #return tf.matmul(X, W) + b
def get_plot(data, title="generated", save_name="save"):
    num_pairs = 16
    fig = plt.figure(frameon=False)
    plt.title(title)
    for pair in range(num_pairs):
        #reshaping to show original test image
        ax = plt.subplot(4,4, pair + 1)
        ax.set_axis_off()
        x_image = data[pair, :, :].T
        ax.matshow(x_image, interpolation='nearest',aspect='auto',cmap="viridis",origin='lower')

    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    
    fig.savefig(save_name + title + '.png', format="png")
    plt.close()

    
def plot_test(sess, train_step, mu, sigma, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(train_step)))

    noise = tf.random_normal([1, hp.latent_dim_1])
    max_len = np.max((mu.shape[1], sigma.shape[1]))
    pad_mu = tf.pad(mu, [[0, 0], [0, max_len-mu.shape[1]], [0, 0]], 'CONSTANT')
    pad_sigma = tf.pad(mu, [[0, 0], [0, max_len-mu.shape[1]], [0, 0]], 'CONSTANT')
    # z_1 is the fisrt leverl output(latent variable) of our Encoder
    z = pad_mu + tf.multiply(noise, tf.exp(0.5*pad_sigma))
    recon = decoder(z, is_training=False, reuse=True)
    recon = sess.run(recon)

    ######## Plot the reconstruction given random latent vector
    title="Generated spectrogram"
    np.savez(save_name + title + ".npz", recon=recon)
    get_plot(recon, title=title, save_name=save_name)
    
    

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
def encoder(encoder_inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Param:
        encoder_inputs:A 2d tensor with shape of [N, T], dtype of int32. N: batch_size  T: real length
        latent_dim:
        h_dim
        '''
    with tf.variable_scope(scope, reuse=reuse):
        # layer 1
        #W_enc_1 = weight_variables([n_pixels, h_dim_1], "W_enc_1")
        #b_enc_1 = bias_variable([h_dim_1], "b_enc_1")
        ## tanh - activation function        avoid vanishing gradient in generative models
        #h_enc_1 = tf.nn.tanh(FC_Layer(encoder_inputs, W_enc_1, b_enc_1))
    
        h_enc_1 = tf.contrib.layers.fully_connected(
                                encoder_inputs,
                                hp.h_dim_1,
                                activation_fn=tf.nn.tanh,
                                biases_initializer=tf.zeros_initializer(),
                                reuse=reuse,
                                scope="encode_lay1")
        # layer 2   Output mean and std of the latent variable distribution
        #W_mu_1 = weight_variables([h_dim_1, latent_dim_1], "W_mu_1")
        #b_mu_1 = bias_variable([latent_dim_1], "b_mu_1")
        #mu_1 = FC_Layer(h_enc_1, W_mu_1, b_mu_1)
        mu_1 = tf.contrib.layers.fully_connected(
                                h_enc_1,
                                hp.latent_dim_1,
                                activation_fn=tf.identity,
                                biases_initializer=tf.zeros_initializer(),
                                scope="en_mu_1")

        #W_sigma_1 = weight_variables([h_dim_1, latent_dim_1], "W_sigma_1")
        #b_sigma_1 = bias_variable([latent_dim_1], "b_sigma_1")
        #sigma_1 = FC_Layer(h_enc_1, W_sigma_1, b_sigma_1)
        sigma_1 = tf.contrib.layers.fully_connected(
                                h_enc_1,
                                hp.latent_dim_1,
                                activation_fn=tf.identity,
                                biases_initializer=tf.zeros_initializer(),
                                scope="en_sigma_1")
        # Reparameterize import Randomness
        noise = tf.random_normal([1, hp.latent_dim_1])
        # z_1 is the fisrt leverl output(latent variable) of our Encoder
        z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))

        ## second level---------------------------- layer 1
        ##W_enc_2 = weight_variables([latent_dim_1, h_dim_2], "W_enc_2")
        ##b_enc_2 = bias_variable([h_dim_2], "b_enc_2")
        ### tanh - activation function        avoid vanishing gradient in generative models
        ##h_enc_2 = tf.nn.tanh(FC_Layer(z_1, W_enc_2, b_enc_2))
        #h_enc_2 = tf.contrib.layers.fully_connected(
                                #z_1,
                                #hp.h_dim_2,
                                #activation_fn=tf.nn.tanh,
                                #biases_initializer=tf.truncated_normal(hp.h_dim_2, stddev=0.1),
                                #name="encode_lay2")
        ## layer 2   Output mean and std of the latent variable distribution
        ##W_mu_2 = weight_variables([h_dim_2, latent_dim_2], "W_mu_2")
        ##b_mu_2 = bias_variable([latent_dim_2], "b_mu_2")
        ##mu_2 = FC_Layer(h_enc_2, W_mu_2, b_mu_2)
        #mu_2 = tf.contrib.layers.fully_connected(
                                #h_enc_2,
                                #hp.latent_dim_2,
                                #activation_fn=tf.identity,
                                #biases_initializer=tf.truncated_normal(hp.latent_dim_2, stddev=0.1),
                                #name="en_mu_2")

        ##W_sigma_2 = weight_variables([h_dim_2, latent_dim_2], "W_sigma_2")
        ##b_sigma_2 = bias_variable([latent_dim_2], "b_sigma_2")
        ##sigma_2 = FC_Layer(h_enc_2, W_sigma_2, b_sigma_2)
        #sigma_2 = tf.contrib.layers.fully_connected(
                                #h_enc_2,
                                #hp.latent_dim_2,
                                #activation_fn=tf.identity,
                                #biases_initializer=tf.truncated_normal(hp.latent_dim_2, stddev=0.1),
                                #name="en_sigma_2")
        ## Reparameterize import Randomness
        #noise_2 = tf.random_normal([1, hp.latent_dim_2])
        ## z_1 is the ultimate output(latent variable) of our Encoder
        #z_2 = mu_2 + tf.multiply(noise_2, tf.exp(0.5*sigma_2))

        #return mu_1, sigma_1, mu_2, sigma_2, z_2
        return mu_1, sigma_1, z_1

############################ Dencoder ############################
"""Build a generative network parametrizing the likelihood of the data
Args:
z: Samples of latent variables
hidden_size: Size of the hidden state of the neural net
Returns:
bernoulli_logits: logits for the Bernoulli likelihood of the data
"""
def decoder(decoder_input, is_training=True, scope="decoder", reuse=None):
    # layer 1
    #W_dec_2 = weight_variables([latent_dim_2, h_dim_2], "W_dec_2")
    #b_dec_2 = bias_variable([h_dim_2], "b_dec_2")
    ## tanh - decode the latent representation
    #h_dec_2 = tf.nn.tanh(FC_Layer(decoder_input, W_dec_2, b_dec_2))
    with tf.variable_scope(scope, reuse=reuse):
        #h_dec_2 = tf.contrib.layers.fully_connected(
                                    #decoder_input,
                                    #hp.h_dim_2,
                                    #activation_fn=tf.nn.tanh,
                                    #biases_initializer=tf.truncated_normal(hp.h_dim_2, stddev=0.1),
                                    #name="dec_lay2")

        ## layer2 - reconstruction the first leverl latent variables
        ##W_rec_2 = weight_variables([h_dim_2, latent_dim_1], "W_dec_2")
        ##b_rec_2 = bias_variable([latent_dim_1], "b_rec_2")
        ##recon_z1 = tf.nn.sigmoid(FC_Layer(h_dec_2, W_rec_2, b_rec_2)) # ?????
        #recon_z1 = tf.contrib.layers.fully_connected(
                                    #h_dec_2,
                                    #hp.latent_dim_1,
                                    #activation_fn=tf.nn.sigmoid,
                                    #biases_initializer=tf.truncated_normal(hp.latent_dim_1, stddev=0.1),
                                    #name="recon_z1")

        ## layer 1
        ##W_dec_1 = weight_variables([latent_dim_1, h_dim_1], "W_dec")
        ##b_dec_1 = bias_variable([h_dim_1], "b_dec")
        ### tanh - decode the latent representation

        ###ipdb.set_trace()
        #noise = tf.random_normal([1, hp.latent_dim_1])
        ## z_1 is the fisrt leverl output(latent variable) of our Encoder
        #z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))
        #residual_z1 = tf.identity(z_1) + recon_z1
        #h_dec_1 = tf.nn.tanh(FC_Layer(residual_z1 , W_dec_1, b_dec_1))
        h_dec_1 = tf.contrib.layers.fully_connected(
                                    decoder_input,
                                    hp.h_dim_1,
                                    activation_fn=tf.nn.tanh,
                                    biases_initializer=tf.zeros_initializer(),
                                    scope="dec_lay1")
        # layer2 - reconstruction the image and output 0 or 1
        #W_rec_1 = weight_variables([h_dim_1, n_pixels], "W_rec_1")
        #b_rec_1 = bias_variable([n_pixels], "b_rec_1")
        ## 784 bernoulli parameter Output
        #reconstruction = tf.nn.sigmoid(FC_Layer(h_dec_1, W_rec_1, b_rec_1))
        reconstruction = tf.contrib.layers.fully_connected(
                                    h_dec_1,
                                    hp.n_mels*hp.r,
                                    activation_fn=tf.nn.sigmoid,
                                    biases_initializer=tf.zeros_initializer(),
                                    scope="reconstruction")

        return reconstruction

def train():

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
    
    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    ################### define graph
    with tf.name_scope('create_inputs'):
        '''################### load data ################'''
        #real and fake image placholders
        encoder_inputs = tf.placeholder(tf.float32, shape = [None, None, hp.n_mels*hp.r], name='encoder_inputs')
        #random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        decoder_input = tf.placeholder(tf.float32, shape = [None, hp.latent_dim_1], name='encoder_inputs')
        is_train = tf.placeholder(tf.bool, name='is_train')

        spec, mag, length, num_batch = get_batch(is_training = True)   

    with tf.name_scope('VAE'):
        ##### encoder
        mu_1, sigma_1, z_1 = encoder(spec)

        ##### decoder
        reconstruction = decoder(z_1)

    ##### Loss
    with tf.name_scope('loss'):
        log_likelihood = tf.reduce_sum(spec * tf.log(reconstruction + 1e-9) + (1 - spec) * tf.log(1 - reconstruction + 1e-9))
        
        KL_divergence = -0.5 * tf.reduce_sum(1 + 2*sigma_1 - tf.pow(mu_1, 2) - tf.exp(2 * sigma_1), reduction_indices=1)
        
        VAE_loss = tf.reduce_mean(log_likelihood - KL_divergence)

        #if hp.target_masking:
            

    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_divergence', tf.reduce_mean(KL_divergence))
    tf.summary.scalar('log_likelihood', tf.reduce_mean(log_likelihood))

    # ### Optimizer
    optimizer = tf.train.AdadeltaOptimizer(0.0005).minimize(- VAE_loss)   # 
    
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    ##################### Set up session and check the glabal save steps
    #saved_global_step = open_sess(args)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #ipdb.set_trace()
    ##################### Saver for storing checkpoints of the model.
    saver = tf.train.Saver(max_to_keep=hp.MAX_TO_KEEP)
    
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
    log_likelihood_array = []
    KL_term_array = []

    iteration_array = [i*hp.CHECK_EVERY for i in range(hp.EPOCHS/hp.CHECK_EVERY)]
    for ii in range(saved_global_step + 1, hp.EPOCHS):
        t1 = time.time()

        # load training batch
        for batch in range(num_batch):
            save_name = result_dir + '/' + hp.VERSION + "_batch{}_".format(batch)
            if batch % 100 == 0:
                print("batch {}/ total {}".format(batch, num_batch))
            spectro = sess.run(spec)

            #run our optimizer on our data
            sess.run(optimizer, feed_dict={encoder_inputs: spectro})
            
            if (batch % 2 == 0):
                summary, vlb_eval, KL_loss, log_loss, _ = sess.run([summaries, VAE_loss, KL_divergence, log_likelihood, optimizer], feed_dict={encoder_inputs: spectro})
                writer.add_summary(summary, step)

                
                #vlb_eval = VAE_loss.eval(session=sess, feed_dict={encoder_inputs: spectro})
                #temp_log = np.mean(log_likelihood1.eval(session=sess, feed_dict={encoder_inputs: spectro}))
                #temp_KL = np.mean(KL_divergence.eval(session=sess, feed_dict={encoder_inputs: spectro}))
                
                #variational_lower_bound_array.append(vlb_eval)  
                #log_likelihood_array.append(temp_log)
                #KL_term_array.append(temp_KL)
                print "Iteration: {}, Loss: {}, log_likelihood: {}, KL_term{}".format(batch, vlb_eval, np.mean(log_loss), np.mean(KL_loss) )
                
            if (batch % hp.CHECK_EVERY == 0):
                mu = sess.run(mu_1)
                sigma = sess.run(sigma_1)

                get_plot(spectro, title="Original spectrogram", save_name=save_name)
                plot_test(sess, batch, mu, sigma, save_name=save_name)

                #plot error
                #iteration_array = [i*CHECK_EVERY for i in range(batch+1)]
                
        
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
            if batch % hp.SAVE_EVERY == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step
        plt.figure()
        #ipdb.set_trace()
        np.savez("losses_file.npz", vaeloss=variational_lower_bound_array, KL=KL_term_array, logloss=log_likelihood_array)
        plt.plot(variational_lower_bound_array, "b*-")
        plt.plot(KL_term_array, "co-")
        plt.plot(log_likelihood_array, "md-")
        plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], loc="best")
        plt.title('Loss per iteration')
        plt.savefig(save_name+"_{}_loss.png".format(batch), format="png")
        plt.close()
    
        #sess.reset(process_data, [audio_batch, test_batch, samples_num])        
        coord.request_stop()
        coord.join(threads)

    
    



if __name__ == "__main__":

    train()

