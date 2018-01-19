# source code adapted from Siraj 

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time #lets clock training time..
import datetime
import os
#from imageio import imwrite
#import data
#from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets    #read data
import ipdb


save_every = 25000         #    2000       # 
plot_every = 5000         #100          #
num_iterations = 1000001   #500       # 
print_every = 2000    # 100   #
end2end = True
version = "MNIST_2level-end2endLoss"

data_dir = "MNIST_data"
n_pixels = 28 * 28

results_dir = 'results/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
logdir = 'model/' + version + "/{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# HyperParameters
latent_dim_1 = 100
h_dim_1 = 500  # size of network
latent_dim_2 = 10
h_dim_2 = 100  # size of network
# load data
mnist = read_data_sets(data_dir, one_hot=True)

# input the image
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))
#Z = tf.placeholder(tf.float32, shape=([None, latent_dim]))

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

def plot_test(model_No, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))

    num_pairs = 10
    image_indices = np.random.randint(0, 200, num_pairs)
    #Lets plot 10 digits
    
    for pair in range(num_pairs):
        #reshaping to show original test image
        x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
        x_image = np.reshape(x, (28,28))
        
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(x_image, aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, 27])
        plt.ylim([27, 0])
        
        #reconstructed image, feed the test image to the decoder
        x_reconstruction = reconstruction.eval(feed_dict={X: x})
        #reshape it to 28x28 pixels
        x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
        #plot it!
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, 27])
        plt.ylim([27, 0])
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
    canvas = np.empty((28 * ny, 28 * nx))
    noise = tf.random_normal([1, 20])
    z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    
    #ipdb.set_trace()
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        z[0:2] = np.array([[xi, yi]])
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
    
        canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
               28:(j + 1) * 28] = reconstruction[0].reshape(28, 28)
    plt.savefig(os.path.join(logdir,
                        'prior_predictive_map_frame_%d.png' % model_No), format="jpg")   # canvas
                        
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

# Loss function = reconstruction error + regularization(similar image's latent representation close)
# X and the reconstruction
if end2end:
    log_likelihood1 = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9))

    KL_divergence1 = -0.5 * tf.reduce_sum(1 + 2*logstd_2 - tf.pow(mu_2, 2) - tf.exp(2 * logstd_2), reduction_indices=1)

    VAE_loss = tf.reduce_mean(log_likelihood1 - KL_divergence1)
else:
    log_likelihood1 = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9))

    KL_divergence1 = -0.5 * tf.reduce_sum(1 + 2*logstd_1 - tf.pow(mu_1, 2) - tf.exp(2 * logstd_1), reduction_indices=1)

    # latent z1 and reconstruced latent z1
    log_likelihood2 = tf.reduce_sum(z_1 * tf.log(recon_z1 + 1e-9) + (1 - z_1) * tf.log(1 - recon_z1 + 1e-9))

    KL_divergence2 = -0.5 * tf.reduce_sum(1 + 2*logstd_2 - tf.pow(mu_2, 2) - tf.exp(2 * logstd_2), reduction_indices=1)

    VAE_loss1 = tf.reduce_mean(log_likelihood1 - KL_divergence1)
    VAE_loss2 = tf.reduce_mean(log_likelihood2 - KL_divergence2)
    VAE_loss = VAE_loss1 + VAE_loss2

optimizer = tf.train.AdadeltaOptimizer(0.0005).minimize(- VAE_loss)   # AdadeltaOptimizer

# Training
#init all variables and start the session!
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
## Add ops to save and restore all the variables.
saver = tf.train.Saver()


#store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array1 = []
log_likelihood_array2 = []
KL_term_array1 = []
KL_term_array2 = []
iteration_array = [i*print_every for i in range(num_iterations/print_every)]

for ii in range(num_iterations):
    
    save_name = results_dir + '/' + "step{}_".format(ii)
    # np.round to make MNIST binary
    #get first batch (200 digits)
    x_batch = np.round(mnist.train.next_batch(200)[0])
    #run our optimizer on our data
    sess.run(optimizer, feed_dict={X: x_batch})
    if (ii % 10 == 0):
        #every 1K iterations record these values
        vlb_eval = VAE_loss.eval(feed_dict={X: x_batch})
        print "Iteration: {}, Loss: {}".format(ii, vlb_eval)
        variational_lower_bound_array.append(vlb_eval)
        log_likelihood_array1.append(np.mean(log_likelihood1.eval(feed_dict={X: x_batch})))
        KL_term_array1.append(np.mean(KL_divergence1.eval(feed_dict={X: x_batch})))
        #log_likelihood_array2.append(np.mean(log_likelihood1.eval(feed_dict={X: x_batch})))
        #KL_term_array2.append(np.mean(KL_divergence2.eval(feed_dict={X: x_batch})))
    t1 = time.time()
    
    if (ii % save_every == 0):
        t2 = time.time()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        saver.save(sess, logdir + '/' + str(ii))
        print("Time for every {} interations is {}".format(save_every, (t2 - t1)))
        

    if (ii % plot_every == 0):
        #plot_prior(ii)
        plot_test(ii, save_name=save_name)

        # plot posterior predictive space
        # Get fixed MNIST digits for plotting posterior means during training
        
        np_x_fixed, np_y = mnist.test.next_batch(100)
        np_x_fixed = np_x_fixed.reshape(100, n_pixels)
        np_x_fixed = (np_x_fixed > 0.5).astype(np.float32)
        np_q_mu = sess.run(mu_1, {X: np_x_fixed})
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
        canvas = np.empty((28 * ny, 28 * nx))
        for ii, yi in enumerate(x_values):
          for j, xi in enumerate(y_values):
            np_z = np.expand_dims(np.append(np.ones((8)), np.array([[xi, yi]])), axis=0)
            #ipdb.set_trace()
            x_mean = sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
            canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
                   28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
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

