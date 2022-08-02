import _pickle
from functools import partial
from pathlib import Path
from typing import TypedDict
from sklearn.preprocessing import QuantileTransformer

from audio.util import *



class Config(TypedDict):
    duration: float
    sample_rate: int
    n_fft: int
    hop_length: int


config: Config = dict(duration=10, sample_rate=22050, n_fft=1024, hop_length=512)

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
              .filter(partial(filter_func, genre="christianrap"), **map_params)
              .cast_column("audio", datasets.Audio(sampling_rate=sr, decode=False))
              .map(partial(load_audio, sampling_rate=sr, remove_silence=True, duration=10), **map_params)
              .map(partial(create_spectrogram, n_fft=config["n_fft"], hop_length=config["hop_length"]), **map_params)
              .map(crop_spectrogram, **map_params)
              .map(partial(normalize_spectrogram, normalizers=normalizers), **map_params)
              .cast_column("audio_spectrogram", datasets.Array3D((n_fft // 2, 431, 2), "float32"))
              .rename_column("audio_spectrogram", "image")
              )

    train_df = subset.to_tf_dataset(batch_size=2, columns=["id", "image"])
    samples = iter(train_df).__next__()
    print(samples)
    # (seq([samples])
    #  .map(lambda x: x["image"])
    #  .map(partial(invert_normalization, normalizers=normalizers))
    #  .map(partial(invert_spectrogram, n_fft=n_fft, hop_length=config["hop_length"]))
    #  .for_each(lambda x: soundfile.write("batch.wav", rearrange(x, "b t -> (b t)"), samplerate=sr))
    #  )
