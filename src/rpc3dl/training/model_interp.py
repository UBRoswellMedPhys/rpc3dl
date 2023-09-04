# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:47:17 2023

@author: johna

Experimenting with generating SHAP model interpretability tools.
"""

import os
import sys
import re
import shap

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from DataGenerator import InputGenerator_v2
import pt_char_lookups

if len(sys.argv) > 1:
    CHECKPOINT_DIR = sys.argv[1]
else:
    CHECKPOINT_DIR = r'D:\model_checkpoints\official\23-08-26_1107'
    
class ShapInterpreter:
    def load_session(self,path):
        self.path = path
        if not hasattr(self,'gen'):
            # this check allows manual config/attach of gen prior to call
            self.setup_generator(path)
        self.load_model(path)
        
    
    def setup_generator(self,path=None):
        if hasattr(self,'path'):
            path = self.path
        if 'datagen_config.json' in os.listdir(path):
            print('Generator configuration found, loading from config...')
            self.gen = InputGenerator_v2.from_config(
                os.path.join(path,'datagen_config.json')
                )
        else:
            raise Exception('No generator config found, try manually configuring.')
            
        active_categories = [
            field for field,val in self.gen.pt_char_settings.items() \
                if val is True
            ]
        print("Active categories: {}".format(active_categories))
    
    # manual config
    """
            DATA_DIR = r"E:\newdata"
            TIME_WINDOW = 'early'
            BATCH_SIZE = 20
        
            active = ['Physical','QOL Baseline']
            active_fields = []
            for group in active:
                active_fields += pt_char_lookups.groups[group]
            PT_CHAR_SETTINGS = {field : True for field in active_fields}
            
            self.gen = InputGenerator_v2(DATA_DIR,time='early',ipsicontra=False)
            self.gen.build_encoders()
            self.gen.pt_char_settings.update(PT_CHAR_SETTINGS)
            self.gen.build_splits(42,val=0.1,test=0.1)
            self.gen.batch_size = BATCH_SIZE
    """

    def load_model(self,path=None):
        if hasattr(self,'path'):
            path = self.path
        if os.path.isdir(path):
            print("Path is a directory, searching for best model...")
            modelfiles = [file for file in os.listdir(path) if file.startswith("model")]
            filenameregex = re.compile(r'model\.(\d*)-loss_(\d*\.\d*)-auc_(\d*\.\d*).h5')
            best_auc = 0
            bestfile = ""
            for file in modelfiles:
                m = re.match(filenameregex,file)
                check_auc = float(m.group(3))
                if check_auc > best_auc:
                    best_auc = check_auc
                    bestfile = file
            print("File {} selected as best model...".format(bestfile))
            to_load = os.path.join(path,bestfile)
        elif os.path.isfile(path):
            print("Path is a file, loading it directly...")
            to_load = path

        self.model = keras.models.load_model(to_load,compile=True)

        # Separate encoder model - this model has 128 layers
        self.encoder = keras.Model(
            inputs=self.model.input[0],
            outputs=self.model.layers[-6].output
            )
        
        self.predictor = keras.Model(
            inputs=self.model.layers[-2].input,
            outputs=self.model.output
            )
    
    def prep_data(self):
        # Now we need to get data prepped for SHAP
        print("Model loaded. Prepping data for SHAP eval...")
        self.bg_data = []
        for i in range(0,len(self.gen.train),50):
            backstop = min(len(self.gen.train),i+50)
            seed_data = [
                self.gen.load_patient(x) for x in self.gen.train[i:backstop]
                ]
            img_embedding = self.encoder.predict(
                np.stack([x[0] for x in seed_data],axis=0)
                )
            batch_data = [
                np.concatenate([img,seed[1]]) for img,seed in \
                    zip(img_embedding,seed_data)
                ]
            self.bg_data += batch_data
        self.bg_data = [np.array(self.bg_data)]
        
        seed_test = [self.gen.load_patient(x) for x in self.gen.test]
        test_embed = self.encoder.predict(np.stack([x[0] for x in seed_test],axis=0))
        self.test_data = [np.concatenate([img,seed[1]]) for img,seed in zip(test_embed,seed_test)]
        self.test_data = [np.array(self.test_data)]

    def generate_shap_values(self):
        
        interp = shap.explainers.Deep(model=self.predictor,data=self.bg_data)
        self.inference = self.predictor.predict(self.test_data[0])
        self.result = interp.shap_values(self.test_data)
        # inference is used later 

    def plot_result(self,figsize=(8,14)):
    
        fig, ax = plt.subplots(1,2,figsize=figsize)
        ax[0].set_yticks(np.arange(self.result[0].shape[0]),labels=self.inference)
        ax[0].set_ylabel('Inference')
        x_desc = self.gen.pt_char_desc()
        ax[0].set_xticks(np.arange(16),labels=[" "]*16)
        ax[0].set_xlabel("Volume Encoded Values")
        ax[1].set_xticks(np.arange(len(x_desc)),labels=x_desc)
        ax[1].set_yticks([])
        im = ax[0].imshow(self.result[0][:,:16],cmap='coolwarm',aspect='auto')
        im2 = ax[1].imshow(self.result[0][:,16:],cmap='coolwarm',aspect='auto')
        cbar = fig.colorbar(im2)
        lim = np.amax(np.abs(self.result[0]))
        im.set_clim(-lim,lim)
        im2.set_clim(-lim,lim)
        plt.tight_layout()
        plt.setp(ax[1].get_xticklabels(), rotation=90, ha="right", va="center",
             rotation_mode="anchor")
        plt.subplots_adjust(bottom=0.2)
        if hasattr(self,'path'):
            plt.savefig(os.path.join(self.path,'shap_fig.png'))
        else:
            plt.show()


if __name__ == '__main__':
    timestamps = [
        '23-08-21_0159',
        '23-08-19_1109',
        '23-08-21_1032',
        '23-08-22_0305',
        '23-08-22_1215',
        '23-08-23_0001',
        '23-08-23_1031',
        '23-08-24_0242',
        '23-08-24_1321',
        '23-08-25_0049',
        '23-08-25_1316',
        '23-08-26_0320',
        '23-08-26_1107',
        '23-08-27_0253',
        '23-08-27_1219',
        '23-08-28_0130'
        ]

    for timestamp in timestamps:
        path = r'D:\model_checkpoints\official\{}'.format(timestamp)
        x = ShapInterpreter()
        x.load_session(path)
        x.prep_data()
        x.generate_shap_values()
        x.plot_result()