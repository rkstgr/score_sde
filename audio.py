from functools import partial
from pathlib import Path
from typing import TypedDict, Any, Dict, List

import librosa
import numpy as np
from jax import numpy as jnp
from audax.core import functional as F
from functional import seq
from sklearn.preprocessing import QuantileTransformer
from einops import rearrange
import _pickle


def load_normalizers(path):
    """Loads the normalizers serialized under the given path

    Args:
        path: Path to serialized quantile transformer normalizers

    Returns:
        (dict) real and imag quantile normalizers
    """
    with open(path, "rb") as f:
        params = _pickle.load(f)

    def load_normalizer(params, key):
        normalizer = QuantileTransformer()
        normalizer.set_params(**params[key]["params"])
        normalizer.quantiles_ = params[key]["quantiles"]
        normalizer.references_ = params[key]["references"]
        return normalizer

    return {
        "real": load_normalizer(params, "real"),
        "imag": load_normalizer(params, "imag")
    }


class Config(TypedDict):
    duration: float
    sample_rate: int
    n_fft: int
    hop_length: int


config: Config = dict(duration=10, sample_rate=22050, n_fft=1024, hop_length=512)


def filter_func(samples: Dict[str, List]) -> List[bool]:
    return ["christianrap" in x for x in samples["genres"]]


def load_audio(samples, sampling_rate=22050, remove_silence=True, duration=10):
    """

    Args:
        samples: Containing key audio, cast to Audio(decode=False)
        sampling_rate: Sampling rate
        remove_silence: (bool) Whether to remove leading and trailing silence
        duration: (numeric) Only load up to this much audio (in seconds), return everything if None

    Returns:
        Dict with "audio_array" list of decoded audio signal (amplitude)
    """

    def take_first_few_seconds(y):
        return y[:int(sampling_rate * duration)]

    x = (seq([x["path"] for x in samples["audio"]])
         .map(lambda x: librosa.load(x, duration=duration + 10, sr=sampling_rate)[0])
         .map(lambda x: librosa.effects.trim(x)[0] if remove_silence else x)
         .map(take_first_few_seconds)
         ).to_list()

    print("load", type(x))
    return {
        "audio_array": x
    }


def create_spectrogram(samples, n_fft=1024, hop_length=512):
    """

    Args:
        samples: (dict) samples containing key "audio_array"
        n_fft: number of samples for fourier transformation
        hop_length: hop length for stft

    Returns:
        Dict with "audio_spectrogram": List of spectrograms (samples, freq, time, (real/imag)) computed from "audio_array"

    Note:
        Since pyarrow does not support complex datatypes the real and imaginary parts of the spectogram are split and
        differentiated in the last dimension
    """
    Y = np.stack(samples["audio_array"], axis=0)
    X = librosa.stft(Y, n_fft=n_fft, hop_length=hop_length)
    X = np.stack([X.real, X.imag], axis=3).astype("float32")

    print("spec", X.shape, X.dtype)
    return {
        "audio_spectrogram": X
    }


def normalize_spectrogram(samples, normalizers):
    """

    Args:
        samples:
        normalizers: fitted scipy transformation, usually Quantile

    Returns:

    """
    X = np.asarray(samples["audio_spectrogram"])

    X_real = normalizers["real"].transform(rearrange(X[:, :, :, 0], "n f t -> (n t) f"))
    X_imag = normalizers["imag"].transform(rearrange(X[:, :, :, 1], "n f t -> (n t) f"))

    X_real = rearrange(X_real, "(n t) f -> n f t", n=X.shape[0])
    X_imag = rearrange(X_imag, "(n t) f -> n f t", n=X.shape[0])

    X = np.stack([X_real, X_imag], axis=3).astype("float32")

    print("norm", X.shape, X.dtype)
    return {
        "audio_spectrogram": X
    }


def invert_normalization(samples, normalizers):
    Xs_real = normalizers["real"].inverse_transform(rearrange(samples[:, :, :, 0], "n f t -> (n t) f"))
    Xs_imag = normalizers["imag"].inverse_transform(rearrange(samples[:, :, :, 1], "n f t -> (n t) f"))

    Xs_real = rearrange(Xs_real, "(n t) f -> n f t", n=samples.shape[0])
    Xs_imag = rearrange(Xs_imag, "(n t) f -> n f t", n=samples.shape[0])
    return np.stack([Xs_real, Xs_imag], axis=3)


def invert_spectrogram(samples, n_fft, hop_length):
    samples = samples[:, :, :, 0] + 1j * samples[:, :, :, 1]
    Ys = librosa.istft(samples, n_fft=n_fft, hop_length=hop_length)
    return Ys


if __name__ == '__main__':
    import datasets
    import warnings
    import soundfile

    warnings.filterwarnings('ignore')

    normalizers = load_normalizers(Path("./audio/normalizers.pckl").resolve())

    mtg = datasets.load_dataset(path="/Volumes/Black T5/dataset/mtg-jamendo/mtg_jamendo.py",
                                cache_dir="/Volumes/Black T5/dataset/huggingface_cache",
                                split=datasets.Split.TRAIN,
                                ignore_verifications=True)

    sr = 22050
    map_params = dict(
        batched=True,
        batch_size=2,
        num_proc=2
    )
    n_fft = config["n_fft"]
    subset = (mtg
              .filter(filter_func, **map_params)
              .cast_column("audio", datasets.Audio(sampling_rate=sr, decode=False))
              .map(partial(load_audio, sampling_rate=sr, remove_silence=True, duration=10), **map_params)
              .map(partial(create_spectrogram, n_fft=config["n_fft"], hop_length=config["hop_length"]), **map_params)
              .map(partial(normalize_spectrogram, normalizers=normalizers), **map_params)
              .cast_column("audio_spectrogram", datasets.Array3D((n_fft // 2 + 1, 431, 2), "float32"))
              .rename_column("audio_spectrogram", "image")
              )

    train_df = subset.to_tf_dataset(batch_size=2, columns=["id", "image"])
    samples = iter(train_df).__next__()
    print(samples)
    (seq([samples])
     .map(lambda x: x["image"])
     .map(partial(invert_normalization, normalizers=normalizers))
     .map(partial(invert_spectrogram, n_fft=n_fft, hop_length=config["hop_length"]))
     .for_each(lambda x: soundfile.write("batch.wav", rearrange(x, "b t -> (b t)"), samplerate=sr))
     )
