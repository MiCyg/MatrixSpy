# import GE2E library
import sys
sys.path.append('lib/ge2e/GE2E')

# jakieś biblioteki hehe
import os
import numpy as np
from pathlib import Path
import scipy
import pickle


# dane trenowania modelu
from encoder.model import SpeakerEncoder
import torch

# wyliczanie embeddingów z nagrania
import librosa
from encoder import audio



# copied from encoder.audio with slight modification
from scipy.spatial.distance import cosine as cds

# live recording
import pyaudio
import keyboard


# ===================== GE2E ======================

# skopiowane z encoder.inference
_model = None  # type: SpeakerEncoder
_device = None  # type: torch.device


def load_model(weights_fpath: Path, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the
    first call to embed_frames() with the default weights file.

    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The
    model will be loaded and will run on this device. Outputs will however always be on the cpu.
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, map_location=torch.device('cpu'))
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))


path_to_models = Path('lib/ge2e/encoder/saved_models')
load_model(path_to_models/'pretrained.pt', device='cpu')


# copied from encoder.inference (import unloads previously loaded model)
def embed_frames_batch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram.

    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    framess = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(framess).detach().cpu().numpy()
    return embed


def count_embedding(sig, fs):
    sig = audio.preprocess_wav(sig, fs)  # resamples, normalizes and removes silence
    fr = audio.wav_to_mel_spectrogram(sig)
    embedding = embed_frames_batch(fr[None, ...])[0]
    return embedding


def which_person(embedding, embeddings):
    pass


# ================= WCZYTANIE NAGRAŃ ==============

def insert_speakers(path_to_records, path_to_save, save_filename='embs', end_filename='_train', verbose=0):
    filenames = [rec for rec in os.listdir(path_to_records) if rec.endswith('.wav')]
    if verbose >= 1: print('rec %s' % filenames)
    audiofiles = {}
    for name in filenames:
        x, fs = librosa.load(path=(path_to_records / name), sr=None, mono=True)
        audiofiles[name] = (fs, x)
    # liczenie/odczytanie embeddingów z próbek głosu

    emb_files = os.listdir(path_to_save)

    eks = {}
    if not save_filename in emb_files:
        for filename in filenames:
            # tylko pliki treningowe
            if filename.endswith(end_filename + '.wav'):
                fs, x = audiofiles[filename]
                if verbose >= 1: print(filename, "Częstotliwość próbkowania: %dHz" % fs)
                eks[filename] = count_embedding(x, fs)

        with open(path_to_save / save_filename, 'wb') as f:
            pickle.dump(eks, f)
    else:
        if verbose >= 1: print('wczytywanie z zapisanego embeddingu...')
        with open(path_to_save / save_filename, 'rb') as f:
            eks = pickle.load(f)

    return eks


# ================ WERYFIKACJA MÓWCY ==============
def softmax(input_dict, min_propability=False):
    # robimy prawdopodobieństwo bayesowskie:
    if min_propability:
        omega = -np.array(list(input_dict.values()))
    else:
        omega = np.array(list(input_dict.values()))

    omega = np.exp(omega)
    omega = np.sum(omega)

    # print('\nmówca:', speaker)
    bayes_prop = {}
    sum_prop = []
    for sp in input_dict:
        if min_propability:
            x = np.exp(-input_dict[sp]) / omega
        else:
            x = np.exp(input_dict[sp]) / omega
        bayes_prop[sp] = x
        sum_prop.append(x)

    return bayes_prop


def tnorm(input_dict, use_std=True):
    out_tnorm = {}

    for name in input_dict:
        values_without_name = list(input_dict.values())
        values_without_name.remove(input_dict[name])

        input_mean = np.mean(values_without_name)
        input_std = np.std(values_without_name)


        if use_std:
            if input_std == 0:
                out_tnorm = None
            else:
                out_tnorm[name] = (input_dict[name] - input_mean) / input_std
        else:
            out_tnorm[name] = (input_dict[name] - input_mean)

    return out_tnorm


def fading(sig, fs, timefold=0.05):
    sig = np.array(sig)
    N = len(sig)
    dt = 1 / fs
    T = N * fs

    out = None
    if T > (2 * timefold):
        N_timefold = round(timefold * fs)

        fold_increase = np.linspace(0, N_timefold, num=N_timefold)
        fold_increase = fold_increase / len(fold_increase)
        len_fold_increase = len(fold_increase)

        fold_decrease = np.linspace(N_timefold, 0, num=N_timefold)
        fold_decrease = fold_decrease / len(fold_increase)
        len_fold_decrease = len(fold_decrease)

        ones_len = N - len_fold_increase - len_fold_decrease

        win = np.append(fold_increase, np.ones(ones_len))
        win = np.append(win, fold_decrease)

        out = win * sig
    else:
        out = None

    return out


def speaker_ver(embedding, embeddings_dict, coeff=0.5):
    # porównanie wyników z próbkami głosu
    distances = {}
    for name in embeddings_dict:
        distances[name] = abs(cds(embeddings_dict[name], embedding))

    # normalizacja i wyznaczanie minimum
    distances_norm = tnorm(distances, use_std=True)
    best_filename = min(distances_norm, key=distances_norm.get)

    # generujemy treshold - poziom poniżej którego może być tylko jeden embedding mówcy
    # liczymy std ze znormalizowanych danych i mnożymy przez współczynnik
    treshold = distances_norm[best_filename] + (coeff * np.std(list(distances_norm.values())))

    # funkcja zwróci None gdy poniżej poziomu treshold pojawi się więcej niż 1 osoba
    number_of_persons_under_treshold = 0
    for sp in distances_norm:
        if distances_norm[sp] < treshold:
            number_of_persons_under_treshold += 1

    return_name = None
    if number_of_persons_under_treshold == 1:
        return_name = best_filename


    return return_name



