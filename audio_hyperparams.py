# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
import datetime


class Hyperparams:
    '''Hyper parameters'''
    # mode
    sanity_check = False
    latent_dim_1 = 128
    h_dim_1 = 512  # size of network
    latent_dim_2 = 32
    h_dim_2 = 256  # size of network
    random_dim = 128

    ## check performance
    CHECK_EVERY = 10
    SAVE_EVERY = 50
    PLOT_EVERY = 20
    CHECKPOINT_EVERY = 50   #
    
    #HEIGHT, WIDTH, CHANNEL, n_pixels = 128, 128, 1, 128 * 128
    
    EPOCHS = int(1e3)+1   # int(1e5)
    LEARNING_RATE = 1e-3

    L2_REGULARIZATION_STRENGTH = 0
    SILENCE_THRESHOLD = 0.1    # 0.3
    MAX_TO_KEEP = 20
    METADATA = False
    
    gen_eval_num = 20
    
    VERSION = 'audio'      #newPokemonmnist_pmFSD_pretty_128'newspectrogram' 'OLLO_NO1'
    DATA_ROOT = 'training_data/corpus'
    LOGDIR_ROOT = 'model'
    DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    RESULT_DIR_ROOT = 'results'

    



    # data
    sound_fpath = 'training_data/corpus'
    max_len = 100 if not sanity_check else 30 # maximum length of text
    min_len = 10 if not sanity_check else 20 # minimum length of text
    
    # signal processing
    sr = 16000 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 30 # Number of inversion iterations 
    use_log_magnitude = True # if False, use magnitude
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 1 # Reduction factor. Paper => 2, 3, 5
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    
    # training scheme
    lr = 0.0005 # Paper => Exponential decay
    logdir = "logdir" if not sanity_check else "logdir_s"
    outputdir = 'samples' if not sanity_check else "samples_s"
    batch_size = 16
    num_epochs = 10000 if not sanity_check else 40 # Paper => 2M global steps!
    loss_type = "l2" # Or you can test "l2"
    num_samples = 16
    
    # etc
    num_gpus = 1 # If you have multiple gpus, adjust this option, and increase the batch size
                 # and run `train_multiple_gpus.py` instead of `train.py`.
    target_zeros_masking = False # If True, we mask zero padding on the target, 
                                 # so exclude them from the loss calculation.     

    HEIGHT = 128
    WIDTH = 128
    CHANNEL = 1
    
