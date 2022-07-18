# libraries

import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import librosa
import librosa.display
import json
import subprocess



import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import librosa
import librosa.display
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# ----------------------------------------------------
# -- Plot audio data
# ----------------------------------------------------


def cv2_imshow(img, window_name="window name"):
    cv2.imshow(window_name, img)
    q = cv2.waitKey()
    cv2.destroyAllWindows()
    return q 

def plot_audio(data, sample_rate, yrange=(-1.1, 1.1), ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    t = np.arange(len(data)) / sample_rate
    plt.plot(t, data)
    plt.xlabel('time (s)')
    plt.ylabel('Intensity')
    plt.title(f'Audio with {len(data)} points, and a {sample_rate} sample rate, ')
    plt.axis([None, None, yrange[0], yrange[1]])

def plot_mfcc(mfcc, sample_rate, method='librosa', ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    assert method in ['librosa', 'cv2']
    
    if method == 'librosa':
        librosa.display.specshow(
            mfcc, sr=sample_rate, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCCs features, len = {mfcc.shape[1]}')
        plt.tight_layout()
    
    elif method == 'cv2':
        cv2.imshow("MFCCs features", mfcc)
        q = cv2.waitKey()
        cv2.destroyAllWindows()

def plot_mfcc_histogram(
        mfcc_histogram, bins, binrange, col_divides,
    ): 
    plt.figure(figsize=(15, 5))
    # Plot by plt
    plt.imshow(mfcc_histogram)
    
    # Add notations
    plt.xlabel(f'{bins} bins, {binrange} range, and {col_divides} columns(pieces)')
    plt.ylabel("Each feature's percentage")
    plt.title("Histogram of MFCCs features")
    plt.colorbar()

# ----------------------------------------------------
# -- Plot machine learning results
# ----------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    tmp = unique_labels(y_true, y_pred)
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm



def play_audio(filename=None, data=None, sample_rate=None):
    if filename:
        print("Play audio:", filename)
        subprocess.call(["cvlc", "--play-and-exit", filename])
    else:
        print("Play audio data")
        filename = '.tmp_audio_from_play_audio.wav'
        write_audio(filename, data, sample_rate)
        subprocess.call(["cvlc", "--play-and-exit", filename])

def read_audio(filename, dst_sample_rate=16000, PRINT=False):
    
    if 0: # This takes 0.4 seconds to read an audio of 1 second. But support for more format
        data, sample_rate = librosa.load(filename) 
    else: # This only takes 0.01 seconds
        data, sample_rate = sf.read(filename) 
    
    assert len(data.shape) == 1, "This project only support 1 dim audio."
    
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
        
    if PRINT:
        print("Read audio file: {}.\n Audio len = {:.2}s, sample rate = {}, num points = {}".format(
            filename, data.size / sample_rate, sample_rate, data.size))
    return data, sample_rate


def write_audio(filename, data, sample_rate, dst_sample_rate=16000):
    
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
        
    sf.write(filename, data, sample_rate)
    # librosa.output.write_wav(filename, data, sample_rate)

# Read / Write list
if 0: # by json
    def write_list(filename, data):
        with open(filename, 'w') as f:
            json.dump(data, f)
            # What's in file: [[2, 3, 5], [7, 11, 13, 15]]

    def read_list(filename):
        with open(filename) as f:
            data = json.load(f)
        return data
else:
    def write_list(filename, data):
        with open(filename, 'w') as f:
            for d in data:
                f.write(str(d) + "\n")
            # What's in file: "[2, 3, 5]\n[7, 11, 13, 15]\n"

    def read_list(filename):
        with open(filename) as f:
            with open(filename, 'r') as f:
                data = [l.rstrip() for l in f.readlines()]
        return data

# lib_commons
import numpy as np 
import cv2
import sys, os
import glob
import time

def create_folder(folder):
    print("Creating folder:", folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def get_filenames(folder, file_types=('*.wav',)):
    filenames = []
    
    if not isinstance(file_types, tuple):
        file_types = [file_types]
        
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

def get_dir_names(folder):
    names = [name for name in os.listdir(folder) if os.path.isdir(name)] 
    return names 

def get_all_names(folder):
    return os.listdir(folder)

def change_suffix(s, new_suffix, index=None):
    i = s.rindex('.')
    si = ""
    if index:
        si = "_" + str(index)
    s = s[:i] + si + "." + new_suffix
    return s 

def int2str(num, len):
    return ("{:0"+str(len)+"d}").format(num)

def add_idx_suffix(s, idx): # /data/two.wav -> /data/two_032.wav
    i = s.rindex('.')
    s = s[:i] + "_" + "{:03d}".format(idx) + s[i:]
    return s 

def cv2_image_float_to_int(img):
    img = (img*255).astype(np.uint8)
    row, col = img.shape
    rate = int(200 / img.shape[0])*1.0
    if rate >= 2:
        img = cv2.resize(img, (int(col*rate), int(row*rate)))
    return img

class Timer(object):
    def __init__(self):
        self.t0 = time.time()
    def report_time(self, event="", prefix=""):
        print(prefix + "Time cost of '{}' is: {:.2f} seconds.".format(
            event, time.time() - self.t0
        ))

if __name__=="__main__":
    print(change_suffix("abc.jpg", new_suffix='avi'))
#lib_datasets
''' 

` class AudioClass
wraps up related operations on an audio

` class AudioDataset
a dataset for loading audios and labels from folder, for training by torch

` def synthesize_audio
API to synthesize one audio

'''


if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import matplotlib.pyplot as plt 
from collections import namedtuple
import copy 
from gtts import gTTS
import subprocess
import glob
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, 
                 data_folder="", classes_txt="",
                 files_name=[], files_label=[],
                 transform=None,
                 bool_cache_audio=False,
                 bool_cache_XY=True, # cache features
                 ):
        
        assert (data_folder and classes_txt) or (files_name, files_label) # input either one
        
        # Get all data's filename and label
        if files_name and files_label:
            self.files_name, self.files_label = files_name, files_label
        else:
            func = AudioDataset.load_filenames_and_labels
            self.files_name, self.files_label = func(data_folder, classes_txt)
        self.files_label = torch.tensor(self.files_label, dtype=torch.int64)
        self.transform = transform

        # Cache computed data
        self.bool_cache_audio = bool_cache_audio
        self.cached_audio = {} # idx : audio
        self.bool_cache_XY = bool_cache_XY
        self.cached_XY = {} # idx : (X, Y). By default, features will be cached
        
    @staticmethod
    def load_filenames_and_labels(data_folder, classes_txt):
        # Load classes
        with open(classes_txt, 'r') as f:
            classes = [l.rstrip() for l in f.readlines()]
        
        # Based on classes, load all filenames from data_folder
        files_name = []
        files_label = []
        for i, label in enumerate(classes):
            folder = data_folder + "/" + label + "/"
            
            names = get_filenames(folder, file_types="*.wav")
            labels = [i] * len(names)
            
            files_name.extend(names)
            files_label.extend(labels)
        
        print("Load data from: ", data_folder)
        print("\tClasses: ", ", ".join(classes))
        return files_name, files_label
            
    def __len__(self):
        return len(self.files_name)

    def get_audio(self, idx):
        if idx in self.cached_audio: # load from cached 
            audio = copy.deepcopy(self.cached_audio[idx]) # copy from cache
        else:  # load from file
            filename=self.files_name[idx]
            audio = AudioClass(filename=filename)
            # print(f"Load file: {filename}")
            self.cached_audio[idx] = copy.deepcopy(audio) # cache a copy
        return audio 
    
    def __getitem__(self, idx):
        
        timer = Timer()
        
        # -- Load audio
        if self.bool_cache_audio:
            audio = self.get_audio(idx)
            print("{:<20}, len={}, file={}".format("Load audio from file", audio.get_len_s(), audio.filename))
        else: # load audio from file
            if (idx in self.cached_XY) and (not self.transform): 
                # if (1) audio has been processed, and (2) we don't need data augumentation,
                # then, we don't need audio data at all. Instead, we only need features from self.cached_XY
                pass 
            else:
                filename=self.files_name[idx]
                audio = AudioClass(filename=filename)
        
        # -- Compute features
        read_features_from_cache = (self.bool_cache_XY) and (idx in self.cached_XY) and (not self.transform)
        
        # Read features from cache: 
        #   If already computed, and no augmentatation (transform), then read from cache
        if read_features_from_cache:
            X, Y = self.cached_XY[idx]
            
        # Compute features:
        #   if (1) not loaded, or (2) need new transform
        else: 
            # Do transform (augmentation)        
            if self.transform:
                audio = self.transform(audio)
                # self.transform(audio) # this is also good. Transform (Augment) is done in place.

            # Compute mfcc feature
            audio.compute_mfcc(n_mfcc=12) # return mfcc
            
            # Compose X, Y
            X = torch.tensor(audio.mfcc.T, dtype=torch.float32) # shape=(time_len, feature_dim)
            Y = self.files_label[idx]
            
            # Cache 
            if self.bool_cache_XY and (not self.transform):
                self.cached_XY[idx] = (X, Y)
            
        # print("{:>20}, len={:.3f}s, file={}".format("After transform", audio.get_len_s(), audio.filename))
        # timer.report_time(event="Load audio", prefix='\t')
        return (X, Y)
    
class AudioClass(object):
    def __init__(self, 
                 data=None, sample_rate=None, filename=None,
                 n_mfcc=12):
        if filename:
            self.data, self.sample_rate = read_audio(filename, dst_sample_rate=None)
        elif (len(data) and sample_rate):
            self.data, self.sample_rate = data, sample_rate
        else:
            assert 0, "Invalid input. Use keyword to input either (1) filename, or (2) data and sample_rate"
            
        self.mfcc = None
        self.n_mfcc = n_mfcc # feature dimension of mfcc 
        self.mfcc_image = None 
        self.mfcc_histogram = None
        
        # Record info of original file
        self.filename = filename
        self.original_length = len(self.data)

    def get_len_s(self): # audio length in seconds
        return len(self.data)/self.sample_rate
    
    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()
    
    def resample(self, new_sample_rate):
        self.data = librosa.core.resample(self.data, self.sample_rate, new_sample_rate)
        self.sample_rate = new_sample_rate
        
    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
       
        # Check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc
            
        # Compute
        self.mfcc = compute_mfcc(self.data, self.sample_rate, n_mfcc)
    
    def compute_mfcc_histogram(
            self, bins=10, binrange=(-50, 200), col_divides=5,
        ): 
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        self._check_and_compute_mfcc()
        self.mfcc_histogram = calc_histogram(
            self.mfcc, bins, binrange, col_divides)
        
        self.args_mfcc_histogram = ( # record parameters
            bins, binrange, col_divides,)
        
    def compute_mfcc_image(
            self, row=200, col=400,
            mfcc_min=-200, mfcc_max=200,
        ):
        ''' Convert mfcc to an image by converting it to [0, 255]'''        
        self._check_and_compute_mfcc()
        self.mfcc_img = mfcc_to_image(
            self.mfcc, row, col, mfcc_min, mfcc_max)
    

    # It's difficult to set this threshold, better not use this funciton.
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        ''' Remove the silence at the beginning of the audio data. '''
         
        l0 = len(self.data) / self.sample_rate
        
        func = remove_silent_prefix_by_freq_domain
        self.data, self.mfcc = func(
            self.data, self.sample_rate, self.n_mfcc, 
            threshold, padding_s, 
            return_mfcc=True
        )
        
        l1 = len(self.data) / self.sample_rate
        print(f"Audio after removing silence: {l0} s --> {l0} s")
        
    # --------------------------- Plotting ---------------------------
    def plot_audio(self, plt_show=False, ax=None):
        plot_audio(self.data, self.sample_rate, ax=ax)
        if plt_show: plt.show()
            
    def plot_mfcc(self, method='librosa', plt_show=False, ax=None):
        self._check_and_compute_mfcc()
        plot_mfcc(self.mfcc, self.sample_rate, method, ax=ax)
        if plt_show: plt.show()
        
    def plot_audio_and_mfcc(self, plt_show=False, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        plot_audio(self.data, self.sample_rate, ax=plt.gca())

        plt.subplot(122)
        self._check_and_compute_mfcc()
        plot_mfcc(self.mfcc, self.sample_rate, method='librosa', ax=plt.gca())

        if plt_show: plt.show()
        
    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()
            
        plot_mfcc_histogram(
            self.mfcc_histogram, *self.args_mfcc_histogram)
        if plt_show: plt.show()

    def plot_mfcc_image(self, plt_show=False):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")
        if plt_show: plt.show()

    # --------------------------- Input / Output ---------------------------
    def write_to_file(self, filename):
        write_audio(filename, self.data, self.sample_rate)
    
    def play_audio(self):
        play_audio(data=self.data, sample_rate=self.sample_rate)
        
def synthesize_audio(
        text, sample_rate=16000, 
        lang='en', tmp_filename=".tmp_audio_from_SynthesizedAudio.wav",
        PRINT=False):
        
    # Create audio
    assert lang in ['en', 'en-uk', 'en-au', 'en-in'] # 4 types of acsents to choose
    if PRINT: print(f"Synthesizing audio for '{text}'...", end=' ')
    tts = gTTS(text=text, lang=lang)
    
    # Save to file and load again
    tts.save(tmp_filename)
    data, sample_rate = librosa.load(tmp_filename) # has to be read by librosa, not soundfile
    subprocess.call(["rm", tmp_filename])
    if PRINT: print("Done!")

    # Convert to my audio class
    audio = AudioClass(data=data, sample_rate=sample_rate)
    audio.resample(sample_rate)
    
    return audio

def shout_out_result(
    filename, predicted_label, 
    preposition_word="is",
    cache_folder="data/examples/"):

    if not os.path.exists(cache_folder): # create folder
        os.makedirs(cache_folder)
        
    fname_preword = cache_folder + preposition_word + ".wav" # create file
    if not os.path.exists(fname_preword):
        synthesize_audio(text=preposition_word, PRINT=True
                         ).write_to_file(filename=fname_preword)

    fname_predict =cache_folder + predicted_label + ".wav" # create file
    if not os.path.exists(fname_predict): 
        synthesize_audio(text=predicted_label, PRINT=True
                         ).write_to_file(filename=fname_predict)
        
    play_audio(filename=filename)
    play_audio(filename=fname_preword)
    play_audio(filename=fname_predict)

def get_wav_filenames(path_to_data):
    ''' Only audio data with .wav suffix are supported by this script '''
    if os.path.isdir(path_to_data):
        filenames = glob.glob(path_to_data + "/*.wav")
        assert len(filenames), f"No .wav files in folder: {path_to_data}"
    elif ".wav" in path_to_data:
        filenames = [path_to_data]
    else:
        raise ValueError('Wrong path_to_data. Only .wav file is supported')
    return filenames


if __name__ == "__main__":
    
    def test_Class_AudioData():
        audio = AudioClass(filename="test_data/audio_numbers.wav")
        audio.plot_audio()
        audio.plot_mfcc()
        audio.plot_mfcc_histogram()
        
        plt.show()
        # audio.play_audio()

    def test_synthesize_audio():
        texts = ["hello"]
        texts = read_list("config/classes_kaggle.names")
        for text in texts:
            audio = synthesize_audio(text, PRINT=True)
            audio.play_audio()
            # audio.write_to_file(f"synthesized_audio_{text}.wav")
            audio.write_to_file(f"{text}.wav")
        
    def main():
        # test_Class_AudioData()
        test_synthesize_audio()

    main()
# lib_augment
''' Data augmentation on audio.
Written in the form of a set of Classes. 
'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa

# ----------------------------------------------------------------------

def rand_uniform(bound, size=None):
    l, r = bound[0], bound[1]
    return np.random.uniform(l, r, size=size)

def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)

def to_tuple(val, left_bound=None):
    if isinstance(val, tuple):
        return val 
    if isinstance(val, list):
        return (val[0], val[1])
    else:
        if left_bound is None:
            left_bound = -val
            assert val>=0, "should be >0, so that (-val, val)"
            return (-val, val)
        else:
            return (left_bound, val)

def random_crop(arr, N): 
    # crop a subarray from array, which has an exact length of N.
    # If len(arr)<N, arr will be duplicated first to make the length > N
    n = len(arr)
    if n < N:
        arr = np.tile(arr, 1+(N//n))
    n = len(arr) # e.g. n=10, N=9, n-N=1, left=[0, 1]=0, right=[9, 10]
    left = np.random.randint(n - N + 1)
    right = left + N
    return arr[left:right]
    ''' Test case:
    for i in range(5):
        x = np.arange(10)
        print(random_crop(x, 10))
    '''
            
class Augmenter(object):
    ''' A wrapper for a serials of transformations '''
    
    def __init__(self, transforms, prob_to_aug=1):
        self.transforms = transforms 
        self.prob_to_aug = prob_to_aug
        
    def __call__(self, audio):
        if np.random.random()>self.prob_to_aug:
            return audio # not augment. direct return.
        else:
            for transform in self.transforms:
                audio = transform(audio)
            return audio 

    # Add simple noise to audio
    class SimpleNoise(object):
        def __init__(self, intensity=(-0.1, 0.1)):
            self.intensity = to_tuple(intensity)
            
        def __call__(self, audio):
            data = audio.data
            noise = rand_uniform(self.intensity, size=data.shape)
            data = data + noise 
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio

    # Add noise to audio, where noises are loaded from file and normalized
    class Noise(object):
        def __init__(self, noise_folder, prob_noise=0.5, intensity=(0, 0.5)):
            self.intensity = to_tuple(intensity)
            
            # Load noises that we will use
            fnames = get_filenames(noise_folder)
            noises = []
            for name in fnames:
                noise, rate = read_audio(filename=name)
                noise = librosa.util.normalize(noise) # normalize noise
                noise = self.repeat_pad_to_time(noise, rate, time=10)
                noises.append(noise)
            self.noises = noises
            self.prob_noise = prob_noise
            
        def __call__(self, audio):
            if np.random.random() > self.prob_noise: # no noise
                return audio
            
            data = audio.data
            
            # add noise
            noise = self.randomly_pick_a_noise() * rand_uniform(self.intensity)
            data = data + random_crop(noise, len(data))
            data[data>+1] = +1
            data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
        def randomly_pick_a_noise(self):
            i = np.random.randint(len(self.noises))
            return self.noises[i]
        
        def repeat_pad_to_time(self, noise, sample_rate, time):
            # repeat the noise data, to make it longer than time
            N = time * sample_rate
            n = len(noise)
            if n < N:
                noise = np.tile(noise, 1+(N//n))
            return noise
        
        
            

    # Shift audio by some time or ratio (>0, to right; <0, to left)
    class Shift(object):
        def __init__(self, time=None, rate=None, keep_size=False):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate)
            elif time: # shift time = time
                self.time = to_tuple(time)
            else:
                assert 0
            self.keep_size = keep_size
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count shift
            data = audio.data
            assert n < len(data), "Shift amount should be smaller than data length."
            
            # Shift audio data
            if time > 0 or n == 0: # move audio data to right
                data = data[n:]
            else:
                data = data[:-n]
            
            # Add padding
            if self.keep_size:
                z = np.zeros(n)
                if time>0: # pad at left
                    data = np.concatenate((z, data))
                else:
                    data = np.concatenate((data, z))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
         
    
    # Crop out a certain time length 
    class Crop(object):
        def __init__(self, time=None):
            assert isinstance(time, tuple) or isinstance(time, list)
            self.time = to_tuple(time)
            
        def __call__(self, audio):
            time = rand_uniform(self.time) # seconds
            data = audio.data 
            n = abs(int(time * audio.sample_rate)) # length to crop
            
            # crop
            if n < len(data):
                data = random_crop(data, n)

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio

    # Pad zeros randomly at left or right by a time or rate >= 0
    class PadZeros(object):
        def __init__(self, time=None, rate=None):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate, left_bound=0)
            elif time: # shift time = time
                self.time = to_tuple(time, left_bound=0)
            else:
                assert 0
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count padding
            data = audio.data
            
            # Shift audio data
            if np.random.random() < 0.5:
                data = np.concatenate(( data, np.zeros(n, ) ))
            else:
                data = np.concatenate(( np.zeros(n, ), data ))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # PlaySpeed audio by a rate (e.g., longer or shorter)
    class PlaySpeed(object):
        def __init__(self, rate=(0.9, 1.1), keep_size=False):
            assert is_list_or_tuple(rate)
            self.rate = rate
            self.keep_size = keep_size
            
        def __call__(self, audio):
            data = audio.data
            rate = rand_uniform(self.rate)
            len0 = len(data) # record original length
            
            # PlaySpeed
            data = librosa.effects.time_stretch(data, rate)
            
            # Pad
            if self.keep_size:
                if len(data)>len0:
                    data = data[:len0]
                else:
                    data = np.pad(data, (0, max(0, len0 - len(data))), "constant")
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # Amplify audio by a rate (e.g., louder or lower)
    class Amplify(object):
        def __init__(self, rate=(0.2, 2)):
            assert is_list_or_tuple(rate)
            self.rate = to_tuple(rate)
            '''
            Test result: For an audio with a median voice,
            if rate=0.2, I could still here it.
            if rate=2, it becomes a little bit loud.
            '''            
        def __call__(self, audio):
            rate = rand_uniform(self.rate)
            data = audio.data * rate
            if rate > 1: # cutoff (Default range for an audio is [-1, 1]).
                # I've experimented this by amplifying an audio, 
                #   saving it file and load again using soundfile library,
                #   and then I found that abs(voice)>1 was cut off to 1.
                data[data>+1] = +1
                data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio


def test_augmentation_effects():
    #from utils.lib import read_audio, write_audio, play_audio
    import copy

    filename = 'test_data/audio_3.wav'
    output_name = 'test_data/tmp_audio.wav'
    audio = AudioClass(filename=filename)
    print("Audio length = ", audio.get_len_s(), " seconds")
    
    for i in range(5): # Test for several times, to see if RANDOM works
        print(i)
        
        # Set up augmenter
        aug = Augmenter([
            Augmenter.Crop(time=(0.5, 1.3)),
            # Augmenter.PadZeros(time=(0.1, 0.3)),
            # Augmenter.Noise(noise_folder="data/noises/", 
            #                 prob_noise=0.6, intensity=(0.1, 0.4)),
            # Augmenter.Shift(rate=0.5, keep_size=False),
            # Augmenter.PlaySpeed(rate=(1.4, 1.5), keep_size=True),
            # Augmenter.Amplify(rate=(2, 2)),
        ])

        # Augment    
        audio_copy = copy.deepcopy(audio)
        aug(audio_copy)
        audio_copy.play_audio()

    audio = audio_copy
    
    # -- Write to file and play. See if its good
    # audio.plot_audio(plt_show=True)
    # audio.write_to_file(output_name)
    
    # -- Load again
    # audio2 = AudioClass(filename=output_name)
    # audio2.plot_audio(plt_show=True)
    
def main():
    test_augmentation_effects()

if __name__ == "__main__":
    main()
     
# lib_ml
''' Data augmentation on audio.
Written in the form of a set of Classes. 
'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2

import librosa

# Import all my libraries


# ----------------------------------------------------------------------

def rand_uniform(bound, size=None):
    l, r = bound[0], bound[1]
    return np.random.uniform(l, r, size=size)

def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)

def to_tuple(val, left_bound=None):
    if isinstance(val, tuple):
        return val 
    if isinstance(val, list):
        return (val[0], val[1])
    else:
        if left_bound is None:
            left_bound = -val
            assert val>=0, "should be >0, so that (-val, val)"
            return (-val, val)
        else:
            return (left_bound, val)

def random_crop(arr, N): 
    # crop a subarray from array, which has an exact length of N.
    # If len(arr)<N, arr will be duplicated first to make the length > N
    n = len(arr)
    if n < N:
        arr = np.tile(arr, 1+(N//n))
    n = len(arr) # e.g. n=10, N=9, n-N=1, left=[0, 1]=0, right=[9, 10]
    left = np.random.randint(n - N + 1)
    right = left + N
    return arr[left:right]
    ''' Test case:
    for i in range(5):
        x = np.arange(10)
        print(random_crop(x, 10))
    '''
            
class Augmenter(object):
    ''' A wrapper for a serials of transformations '''
    
    def __init__(self, transforms, prob_to_aug=1):
        self.transforms = transforms 
        self.prob_to_aug = prob_to_aug
        
    def __call__(self, audio):
        if np.random.random()>self.prob_to_aug:
            return audio # not augment. direct return.
        else:
            for transform in self.transforms:
                audio = transform(audio)
            return audio 

    # Add simple noise to audio
    class SimpleNoise(object):
        def __init__(self, intensity=(-0.1, 0.1)):
            self.intensity = to_tuple(intensity)
            
        def __call__(self, audio):
            data = audio.data
            noise = rand_uniform(self.intensity, size=data.shape)
            data = data + noise 
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio

    # Add noise to audio, where noises are loaded from file and normalized
    class Noise(object):
        def __init__(self, noise_folder, prob_noise=0.5, intensity=(0, 0.5)):
            self.intensity = to_tuple(intensity)
            
            # Load noises that we will use
            fnames = get_filenames(noise_folder)
            noises = []
            for name in fnames:
                noise, rate = read_audio(filename=name)
                noise = librosa.util.normalize(noise) # normalize noise
                noise = self.repeat_pad_to_time(noise, rate, time=10)
                noises.append(noise)
            self.noises = noises
            self.prob_noise = prob_noise
            
        def __call__(self, audio):
            if np.random.random() > self.prob_noise: # no noise
                return audio
            
            data = audio.data
            
            # add noise
            noise = self.randomly_pick_a_noise() * rand_uniform(self.intensity)
            data = data + random_crop(noise, len(data))
            data[data>+1] = +1
            data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
        def randomly_pick_a_noise(self):
            i = np.random.randint(len(self.noises))
            return self.noises[i]
        
        def repeat_pad_to_time(self, noise, sample_rate, time):
            # repeat the noise data, to make it longer than time
            N = time * sample_rate
            n = len(noise)
            if n < N:
                noise = np.tile(noise, 1+(N//n))
            return noise
        
        
            

    # Shift audio by some time or ratio (>0, to right; <0, to left)
    class Shift(object):
        def __init__(self, time=None, rate=None, keep_size=False):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate)
            elif time: # shift time = time
                self.time = to_tuple(time)
            else:
                assert 0
            self.keep_size = keep_size
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count shift
            data = audio.data
            assert n < len(data), "Shift amount should be smaller than data length."
            
            # Shift audio data
            if time > 0 or n == 0: # move audio data to right
                data = data[n:]
            else:
                data = data[:-n]
            
            # Add padding
            if self.keep_size:
                z = np.zeros(n)
                if time>0: # pad at left
                    data = np.concatenate((z, data))
                else:
                    data = np.concatenate((data, z))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
         
    
    # Crop out a certain time length 
    class Crop(object):
        def __init__(self, time=None):
            assert isinstance(time, tuple) or isinstance(time, list)
            self.time = to_tuple(time)
            
        def __call__(self, audio):
            time = rand_uniform(self.time) # seconds
            data = audio.data 
            n = abs(int(time * audio.sample_rate)) # length to crop
            
            # crop
            if n < len(data):
                data = random_crop(data, n)

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio

    # Pad zeros randomly at left or right by a time or rate >= 0
    class PadZeros(object):
        def __init__(self, time=None, rate=None):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate, left_bound=0)
            elif time: # shift time = time
                self.time = to_tuple(time, left_bound=0)
            else:
                assert 0
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count padding
            data = audio.data
            
            # Shift audio data
            if np.random.random() < 0.5:
                data = np.concatenate(( data, np.zeros(n, ) ))
            else:
                data = np.concatenate(( np.zeros(n, ), data ))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # PlaySpeed audio by a rate (e.g., longer or shorter)
    class PlaySpeed(object):
        def __init__(self, rate=(0.9, 1.1), keep_size=False):
            assert is_list_or_tuple(rate)
            self.rate = rate
            self.keep_size = keep_size
            
        def __call__(self, audio):
            data = audio.data
            rate = rand_uniform(self.rate)
            len0 = len(data) # record original length
            
            # PlaySpeed
            data = librosa.effects.time_stretch(data, rate)
            
            # Pad
            if self.keep_size:
                if len(data)>len0:
                    data = data[:len0]
                else:
                    data = np.pad(data, (0, max(0, len0 - len(data))), "constant")
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # Amplify audio by a rate (e.g., louder or lower)
    class Amplify(object):
        def __init__(self, rate=(0.2, 2)):
            assert is_list_or_tuple(rate)
            self.rate = to_tuple(rate)
            '''
            Test result: For an audio with a median voice,
            if rate=0.2, I could still here it.
            if rate=2, it becomes a little bit loud.
            '''            
        def __call__(self, audio):
            rate = rand_uniform(self.rate)
            data = audio.data * rate
            if rate > 1: # cutoff (Default range for an audio is [-1, 1]).
                # I've experimented this by amplifying an audio, 
                #   saving it file and load again using soundfile library,
                #   and then I found that abs(voice)>1 was cut off to 1.
                data[data>+1] = +1
                data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio


def test_augmentation_effects():
    #from utils.lib import read_audio, write_audio, play_audio
    import copy

    filename = 'test_data/audio_3.wav'
    output_name = 'test_data/tmp_audio.wav'
    audio = AudioClass(filename=filename)
    print("Audio length = ", audio.get_len_s(), " seconds")
    
    for i in range(5): # Test for several times, to see if RANDOM works
        print(i)
        
        # Set up augmenter
        aug = Augmenter([
            Augmenter.Crop(time=(0.5, 1.3)),
            # Augmenter.PadZeros(time=(0.1, 0.3)),
            # Augmenter.Noise(noise_folder="data/noises/", 
            #                 prob_noise=0.6, intensity=(0.1, 0.4)),
            # Augmenter.Shift(rate=0.5, keep_size=False),
            # Augmenter.PlaySpeed(rate=(1.4, 1.5), keep_size=True),
            # Augmenter.Amplify(rate=(2, 2)),
        ])

        # Augment    
        audio_copy = copy.deepcopy(audio)
        aug(audio_copy)
        audio_copy.play_audio()

    audio = audio_copy
    
    # -- Write to file and play. See if its good
    # audio.plot_audio(plt_show=True)
    # audio.write_to_file(output_name)
    
    # -- Load again
    # audio2 = AudioClass(filename=output_name)
    # audio2.plot_audio(plt_show=True)
    
def main():
    test_augmentation_effects()

if __name__ == "__main__":
    main()
     
        

import numpy as np 
import sklearn
from collections import OrderedDict, namedtuple
import matplotlib.pyplot as plt 
import types 

def split_train_test(X, Y, test_size=0, USE_ALL=False, dtype='numpy', if_print=True):
    
    assert dtype in ['numpy', 'list']
    
    def _print(s):
        if if_print:
            print(s)
            
    _print("split_train_test:")
    if dtype == 'numpy':
        _print("\tData size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
        if USE_ALL:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            f = sklearn.model_selection.train_test_split
            tr_X, te_X, tr_Y, te_Y = f(X, Y, test_size=test_size, random_state=14123)
    elif dtype == 'list':
        _print("\tData size = {}, feature dimension = {}".format(len(X), len(X[0])))
        if USE_ALL:
            tr_X = X[:]
            tr_Y = Y[:]
            te_X = X[:]
            te_Y = Y[:]
        else:
            N = len(Y)
            train_size = int((1-test_size)*N)
            randidx = np.random.permutation(N)
            n1, n2 = randidx[0:train_size], randidx[train_size:]
            def get(arr_vals, arr_idx):
                return [arr_vals[idx] for idx in arr_idx]
            tr_X = get(X, n1)[:]
            tr_Y = get(Y, n1)[:]
            te_X = get(X, n2)[:]
            te_Y = get(Y, n2)[:]
    _print("\tNum training: {}".format(len(tr_Y)))
    _print("\tNum evaluation: {}".format(len(te_Y)))
    return tr_X, tr_Y, te_X, te_Y

def split_train_eval_test(X, Y, ratios=[0.8, 0.1, 0.1], dtype='list'):
    
    X1, Y1, X2, Y2 = split_train_test(
        X, Y, 
        1-ratios[0], 
        dtype=dtype, if_print=False)
    
    X2, Y2, X3, Y3 = split_train_test(
        X2, Y2, 
        ratios[2]/(ratios[1]+ratios[2]),
        dtype=dtype, if_print=False)
    
    r1, r2, r3 = 100*ratios[0], 100*ratios[1], 100*ratios[2]  
    n1, n2, n3 = len(Y1), len(Y2), len(Y3)
    print(f"Split data into [Train={n1} ({r1}%), Eval={n2} ({r2}%),  Test={n3} ({r3}%)]")
    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = X1, Y1, X2, Y2, X3, Y3 
    return tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y



class TrainingLog(object):
    def __init__(self,
                 training_args=None, # arguments in training
                #  MAX_EPOCH = 1000,
                 ):
        
        if not isinstance(training_args, dict):
            training_args = training_args.__dict__
        self.training_args = training_args 
        
        self.epochs = []
        self.accus_train = []
        self.accus_eval = []
        self.accus_test = []
        
    def store_accuracy(self, epoch, train=-0.1, eval=-0.1, test=-0.1):
        self.epochs.append(epoch)
        self.accus_train.append(train)
        self.accus_eval.append(eval)
        self.accus_test.append(test)
        # self.accu_table[epoch] = self.AccuItems(train, eval, test)
        
    def plot_train_eval_accuracy(self):
        plt.cla()
        t = self.epochs
        plt.plot(t, self.accus_train, 'r.-', label="train")
        plt.plot(t, self.accus_eval, 'b.-', label="eval")
        plt.title("Accuracy on train/eval dataset")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        
        # lim
        # plt.ylim([0.2, 1.05])
        plt.legend(loc='upper left')
        
    def save_log(self, filename):
        with open(filename, 'w') as f:
            
            # -- Args
            f.write("Args:" + "\n")
            for key, val in self.training_args.items():
                s = "\t{:<20}: {}".format(key, val)
                f.write(s + "\n")
            f.write("\n"
                    )
            # -- Accuracies
            f.write("Accuracies:" + "\n")
            f.write("\t{:<10}{:<10}{:<10}{:<10}\n".format(
                "Epoch", "Train", "Eval", "Test"))
            
            for i in range(len(self.epochs)):
                epoch = self.epochs[i]
                train = self.accus_train[i]
                eval = self.accus_eval[i]
                test = self.accus_test[i]
                f.write("\t{:<10}{:<10.3f}{:<10.3f}{:<10.3f}\n".format(
                    epoch, train, eval, test))
                
def test_logger():
    
    # Set arguments 
    args = types.SimpleNamespace()
    args.input_size = 12  
    args.weight_decay = 0.00
    args.data_folder = "data/data_train/"

    # Test
    log = TrainingLog(training_args=args)
    log.store_accuracy(1, 0.7, 0.2)
    log.store_accuracy(5, 0.8, 0.3)
    log.store_accuracy(10, 0.9, 0.4)
    log.plot_train_eval_accuracy()
    log.save_log("tmp_from_lib_ml_test_logger.txt")
    plt.show()
    
if __name__ == "__main__":
    test_logger()        




''' Functions for processing audio and audio mfcc features '''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import warnings
import scipy
from scipy import signal

# How long is a mfcc data frame ?        
MFCC_RATE = 50 # TODO: It's about 1/50 s, I'm not sure.

# ----------------------------------------------------------------------
if 1: # Basic maths
    def rand_num(val): # random [-val, val]
        return (np.random.random()-0.5)*2*val
    
    def integral(arr):
        ''' sums[i] = sum(arr[0:i]) '''
        sums = [0]*len(arr)
        for i in range(1, len(arr)):
            sums[i] = sums[i-1] + arr[i]
        return sums
    
    def filter_by_average(arr, N):
        ''' Average filtering a data sequency by window size of N '''
        cumsum = np.cumsum(np.insert(arr, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N  

# ----------------------------------------------------------------------

if 1: # Time domain processings
    
    def resample_audio(data, sample_rate, new_sample_rate):
        data = librosa.core.resample(data, sample_rate, new_sample_rate)
        return data, new_sample_rate

    def filter_audio_by_average(data, sample_rate, window_seconds):
        ''' Replace audio data[j] with np.mean(data[i:j]) '''
        ''' 
        Output:
            audio data with same length
        '''
        
        window_size = int(window_seconds * sample_rate)
        
        if 1: # Compute integral arr, then find interval sum
            sums = integral(data)
            res = [0]*len(data)
            for i in range(1, len(data)):
                prev = max(0, i - window_size)
                res[i] = (sums[i] - sums[prev]) / (i - prev)
        else: # Use numpy built-in
            filter_by_average(data, window_size)
            
        return res


    def remove_silent_prefix_by_freq_domain(
            data, sample_rate, n_mfcc, threshold, padding_s=0.2,
            return_mfcc=False):
        
        # Compute mfcc, and remove silent prefix
        mfcc_src = compute_mfcc(data, sample_rate, n_mfcc)
        mfcc_new = remove_silent_prefix_of_mfcc(mfcc_src, threshold, padding_s)

        # Project len(mfcc) to len(data)
        l0 = mfcc_src.shape[1]
        l1 = mfcc_new.shape[1]
        start_idx = int(data.size * (1 - l1 / l0))
        new_audio = data[start_idx:]
        
        # Return
        if return_mfcc:        
            return new_audio, mfcc_new
        else:
            return new_audio
            
    def remove_silent_prefix_by_time_domain(
            data, sample_rate, threshold=0.25, window_s=0.1, padding_s=0.2):
        ''' Remove silent prefix of audio, by checking voice intensity in time domain '''
        ''' 
            threshold: voice intensity threshold. Voice is in range [-1, 1].
            window_s: window size (seconds) for averaging.
            padding_s: padding time (seconds) at the left of the audio.
        '''
        window_size = int(window_s * sample_rate)
        trend = filter_by_average(abs(data), window_size)
        start_idx = np.argmax(trend > threshold)
        start_idx = max(0, start_idx + window_size//2 - int(padding_s*sample_rate))
        return data[start_idx:]



# ----------------------------------------------------------------------
if 1: # Frequency domain processings (on mfcc)
    
    def compute_mfcc(data, sample_rate, n_mfcc=12):
        # Extract MFCC features
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
        mfcc = librosa.feature.mfcc(
            y=data,
            sr=sample_rate,
            n_mfcc=n_mfcc,   # How many mfcc features to use? 12 at most.
                        # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
        )
        return mfcc 
    
    def compute_log_specgram(audio, sample_rate, window_size=20,
                    step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        MAX_FREQ = 9999999999999
        for i in range(len(freqs)):
            if freqs[i] > MAX_FREQ:
                break 
        freqs = freqs[0:i]
        spec = spec[:, 0:i]
        return freqs, np.log(spec.T.astype(np.float32) + eps)

    def remove_silent_prefix_of_mfcc(mfcc, threshold, padding_s=0.2):
        '''
        threshold:  Audio is considered started at t0 if mfcc[t0] > threshold
        padding: pad data at left (by moving the interval to left.)
        '''
        
        # Set voice intensity
        voice_intensity = mfcc[1]
        if 1:
            voice_intensity += mfcc[0]
            threshold += -100
        
        # Threshold to find the starting index
        start_indices = np.nonzero(voice_intensity > threshold)[0]
        
        # Return sliced mfcc
        if len(start_indices) == 0:
            warnings.warn("No audio satisifies the given voice threshold.")
            warnings.warn("Original data is returned")
            return mfcc
        else:
            start_idx = start_indices[0]
            # Add padding
            start_idx = max(0, start_idx - int(padding_s * MFCC_RATE))
            return mfcc[:, start_idx:]
    
    def mfcc_to_image(mfcc, row=200, col=400,
                    mfcc_min=-200, mfcc_max=200):
        ''' Convert mfcc to an image by converting it to [0, 255]'''
        
        # Rescale
        mfcc_img = 256 * (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
        
        # Cut off
        mfcc_img[mfcc_img>255] = 255
        mfcc_img[mfcc_img<0] = 0
        mfcc_img = mfcc_img.astype(np.uint8)
        
        # Resize to desired size
        img = cv2.resize(mfcc_img, (col, row))
        return img
    
    def pad_mfcc_to_fix_length(mfcc, goal_len=100, pad_value=-200):
        feature_dims, time_len = mfcc.shape
        if time_len >= goal_len:
            mfcc = mfcc[:, :-(time_len - goal_len)] # crop the end of audio
        else:
            n = goal_len - time_len
            zeros = lambda n: np.zeros((feature_dims, n)) + pad_value
            if 0: # Add paddings to both side
                n1, n2 = n//2, n - n//2
                mfcc = np.hstack(( zeros(n1), mfcc, zeros(n2)))
            else: # Add paddings to left only
                mfcc = np.hstack(( zeros(n), mfcc))
        return mfcc
    
    def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides=5): 
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        feature_dims, time_len = mfcc.shape
        cc = time_len//col_divides # cols / num_hist = size of each hist
        def calc_hist(row, cl, cr):
            hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
            return hist/(cr-cl)
        features = []
        for j in range(col_divides):
            row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims) ]
            row_hists = np.vstack(row_hists) # shape=(feature_dims, bins)
            features.append(row_hists)
        features = np.hstack(features)# shape=(feature_dims, bins*col_divides)
        return features 
    

if __name__ == "__main__":
    pass
