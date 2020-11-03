# R-CED based Speech Enhancement:
An Encoder-Decoder based approach to denoise an input speech signal. [Following](https://github.com/pushshift/api) paper's architecture is used in the Assignment.

## Steps:
  - First step was dataset creation. For this cleaned speech signals were taken from [Mozilla Common Voice Dataset](https://www.kaggle.com/mozillaorg/common-voice) and noise was taken from [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
  - Aforementioned datasets were used to create a mixed signal dataset (Noise + Clean Audio). Dataset was converted to ```TFRecord``` format.
  - Convolution Encoder-Decoder based architecture from the aforementioned paper was then used to train the model using STFT magnitude spectrum representations of audio signals.
  
 

## Instructions to run:
  - ```Denoise_Test.ipynb``` demonstrates training process.
  - ```Denoise_Test.ipynb``` demonstrates inference process.
  - ```denoiser_test.py``` takes input clean audio and noisy audio, mix them to create a mixed signal and then apply then apply inference process to produce a denoised output.
  - Ensure that you are using Tensorflow version 2.3.0 and Librosa versions other than 0.8.0
 
  - Execute the following command and output audio file will be generated with the same name as of the input clean audio but with an "_output.wav" suffix.
 
```sh
$ python3 denoiser_test.py <path_to_clean_audio_file> <path_to_noise_audio_file>
```
  - For Example:
 
```sh
$ python3 denoiser_test.py test_data/clean_audio/sample-000000.mp3 test_data/noises/46655-6-0-0.wav
```
## Future Work and Improvements:
  - The current system does not performs very well.
  - Due to computational and storage constraints model was trained on only a part of the aforementioned dataset (5GB), this is the key reason for the mediocre performance of the system.
  - Next step will be to train on as much data as possible.
  - Use different sampling rates.
  - Use other architectures mentioned in the paper (RNNs, DNNs)
  - Dive into GAN and see its performance on the task.
  - Use of different features like MFCC, CQT and others.


    
