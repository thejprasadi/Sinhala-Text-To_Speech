"""mailabs dataset is sampled at 16000 kHz with 0.5 seconds of silence
    in the start and end of the audio data. Make sure to change the
    sample_size hparams to match this.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


def build_from_path(in_dir, out_dir, books, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the mailabs Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the mailabs Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  books = books.strip().split(',')
  print('preprocess these books', books)
  for book in books:
    book_dir = os.path.join(in_dir, book)
    with open(os.path.join(book_dir, 'metadata.csv'), encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('|')
        name = parts[0]
        wav_path = os.path.join(book_dir, 'wavs', '%s.wav' % name)
        # normalized version of text i.e numbers convered to words
        text = parts[2]
        futures.append(
            executor.submit(partial(
                _process_utterance, out_dir, name, wav_path, text)
            ))
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, name, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # trim silences here
  wav = audio.trim_silence(wav)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'mailabs-spec-{}.npy'.format(name)
  mel_filename = 'mailabs-mel-{}.npy'.format(name)
  np.save(os.path.join(out_dir, spectrogram_filename),
          spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename),
          mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
