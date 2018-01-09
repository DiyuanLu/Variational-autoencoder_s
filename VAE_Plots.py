import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time #lets clock training time..
import os
#import data
from tensorflow.examples.tutorials.mnist import input_data
import pdb

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

SAVE_EVERY = 20000
plot_every = 5000
version = "MNIST"
n_pixels = 28 * 28
# input the image
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

def plot_likelihood(model_No, load_model = False, save_name="save"):
    '''
    visualize the prior predictive distribution by looking samples from the likelihood
    Param:
    '''
    model = 
    # z is the ultimate output(latent variable) of our Encoder
    z = mu + tf.multiply(noise, tf.exp(0.5*logstd))

    ############################ Dencoder ############################
    # layer 1
    W_dec = weight_variables([latent_dim, h_dim], "W_dec")
    b_dec = bias_variable([h_dim], "b_dec")
    # tanh - decode the latent representation
    h_dec = tf.nn.tanh(FC_Layer(z, W_dec, b_dec))

    # layer2 - reconstruction the image and output 0 or 1
    W_rec = weight_variables([h_dim, n_pixels], "W_dec")
    b_rec = bias_variable([n_pixels], "b_rec")
    # 784 bernoulli parameter Output
    reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
    
def plot_test(model_No, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), "model/MNIST/{}".format(model_No)))

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
    #pdb.set_trace()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")



# Print progress and save samples every so often
t0 = time.time()
for i in range(FLAGS.n_iterations):
    # Re-binarize the data at every batch; this improves results
    np_x, _ = mnist.train.next_batch(FLAGS.batch_size)
    np_x = np_x.reshape(FLAGS.batch_size, 28, 28, 1)
    np_x = (np_x > 0.5).astype(np.float32)
    sess.run(train_op, {x: np_x})

    # Print progress and save samples every so often
    t0 = time.time()
    if i % FLAGS.print_every == 0:
      np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
      train_writer.add_summary(summary_str, i)
      print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(
          i,
          np_elbo / FLAGS.batch_size,
          FLAGS.batch_size * FLAGS.print_every / (time.time() - t0)))
      t0 = time.time()

      # Save samples
      np_posterior_samples, np_prior_samples = sess.run(
          [posterior_predictive_samples, prior_predictive_samples], {x: np_x})
      for k in range(FLAGS.n_samples):
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_data.jpg' % (i, k))
        imsave(f_name, np_x[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_sample.jpg' % (i, k))
        imsave(f_name, np_posterior_samples[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_prior_predictive_%d.jpg' % (i, k))
        imsave(f_name, np_prior_samples[k, :, :, 0])

      # Plot the posterior predictive space
      if FLAGS.latent_dim == 2:
        np_q_mu = sess.run(q_mu, {x: np_x_fixed})
        cmap = mpl.colors.ListedColormap(sns.color_palette("husl"))
        #f, ax = plt.subplots(1, figsize=(6 * 1.1618, 6))
        #im = ax.scatter(np_q_mu[:, 0], np_q_mu[:, 1], c=np.argmax(np_y, 1), cmap=cmap,
                        #alpha=0.7)
        #ax.set_xlabel('First dimension of sampled latent variable $z_1$')
        #ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
        #ax.set_xlim([-10., 10.])
        #ax.set_ylim([-10., 10.])
        #f.colorbar(im, ax=ax, label='Digit class')
        #plt.tight_layout()
        #plt.savefig(os.path.join(FLAGS.logdir,
                                 #'posterior_predictive_map_frame_%d.png' % i))
        #plt.close()

        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)
        canvas = np.empty((28 * ny, 28 * nx))
        for ii, yi in enumerate(x_values):
          for j, xi in enumerate(y_values):
            np_z = np.array([[xi, yi]])
            x_mean = sess.run(prior_predictive_inp_sample, {z_input: np_z})
            canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
                   28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
        imsave(os.path.join(FLAGS.logdir,
                            'prior_predictive_map_frame_%d.png' % i), canvas)
        # plt.figure(figsize=(8, 10))
        # Xi, Yi = np.meshgrid(x_values, y_values)
        # plt.imshow(canvas, origin="upper")
        # plt.tight_layout()
        # plt.savefig()

  # Make the gifs
  if FLAGS.latent_dim == 2:
    os.system(
        'convert -delay 15 -loop 0 {0}/posterior_predictive_map_frame*png {0}/posterior_predictive.gif'
        .format(FLAGS.logdir))
    os.system(
        'convert -delay 15 -loop 0 {0}/prior_predictive_map_frame*png {0}/prior_predictive.gif'
        .format(FLAGS.logdir))
