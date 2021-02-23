import gpt_2_simple as gpt2
import os
import tensorflow as tf
import time
import json
import shutil
from src.download_model import download

def train(input_file):

    if os.path.exists('models/temp'):
        shutil.rmtree('models/temp')
    
    if os.path.exists('models/124M'):
        pass
    else:
        download()

    sess = gpt2.start_tf_sess()
   
    model_name = '124M'
    model_dir = 'models/'
    training_dir = 'src/training_data/'
    file_name = input_file.split('.')[0]

    gpt2.finetune(sess,
        training_dir+input_file,
        model_name=model_name,
        checkpoint_dir=model_dir+'temp/',
        run_name='',
        steps=1)
    
    gpt2.reset_session(sess)
    
    if os.path.exists('models/latest'):
        shutil.rmtree('models/latest')
    shutil.copytree('models/temp','models/latest')
    # shutil.rmtree('models/temp')