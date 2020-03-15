"""An optional module to post preprocess the text before syntehsizing
"""
import io
import math
from util import audio
import numpy as np

punctuations = ['.', '?', '!']

split_punctuations = [',', '.', '-', '?', '!', ':', ';']

letter_lookup = {
  'A': 'ayy',
  'B': 'bee',
  'C': 'see',
  'D': 'dee',
  'E': 'eee',
  'F': 'eff',
  'G': 'jee',
  'H': 'aitch',
  'I': 'eye',
  'J': 'jay',
  'K': 'kay',
  'L': 'el',
  'M': 'em',
  'N': 'en',
  'O': 'oow',
  'P': 'pee',
  'Q': 'queue',
  'R': 'are',
  'S': 'es',
  'T': 'tee',
  'U': 'you',
  'V': 'vee',
  'W': 'double you',
  'X': 'ex',
  'Y': 'why',
  'Z': 'zee'
}

def replace_acronym(text):
  for idx, word in enumerate(text):
    if "{" in word and "}" in word:
      continue
    if len(word) == 1:
      continue
    if word.isupper():
      sound = ""
      for letter in word.strip():
        if letter_lookup.get(letter):
          sound += letter_lookup.get(letter) + " "
      text[idx] = sound
  return text

def custom_splitter(text):
  if "{" in text and "}" in text:
    acc = []
    split = text.split("}")
    for word in split:
      if "{" in word:
        acc.append(word + "}")
      else:
        acc.append(word)
    return acc
  else:
    return text.split()

def add_punctuation(text):
  if len(text) < 1:
    return text
  if len(text) < 10:
    if text[-1] in punctuations:
      if text[-1] != ".":
        return text[:-1] + "."
  if text[-1] not in punctuations:
    text += '.'
  return text

def break_chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield " ".join(l[i:i + n])

def split_by_threshold(text, threshold):
  text_list = text.split()
  
  if len(text_list) <= threshold:
    return [text]

  if threshold < len(text_list) < (threshold*2):
    return list(break_chunks(
        text_list,
        int(math.ceil(len(text_list) / 2))
    ))
  elif (threshold*2) < len(text_list) < (threshold*3):
    return list(break_chunks(
        text_list,
        int(math.ceil(len(text_list) / 3))
    ))
  elif (threshold*3) < len(text_list) < (threshold*4):
    return list(break_chunks(
        text_list,
        int(math.ceil(len(text_list) / 4))
    ))
  else:
    return list(break_chunks(
        text_list,
        int(math.ceil(len(text_list) / 4))
    ))

def synthesize_helper(text, synthesizer, threshold=10):
    text_list = text.split()
    if len(text_list) <= threshold*1.3:
      text = " ".join(replace_acronym(text_list))
      print(text.encode('utf-8'))
      wav, _ = synthesizer.synthesize(add_punctuation(text), return_wav=True)
      out = io.BytesIO()
      audio.save_wav(wav, out)
      return out.getvalue()

    split_by_punc = None
    if len(text_list) >= threshold:
      for punc in split_punctuations:
        if punc in text:
          split_by_punc = text.split(punc)
          break

    chunks = []
    if split_by_punc:
      for sentence in split_by_punc:
        sentence = sentence.strip()
        chunk = split_by_threshold(sentence, threshold)
        chunks += split_by_threshold(sentence, threshold)
    else:
      chunks += split_by_threshold(text, threshold)

    combined_wav = np.array([])
    for idx, chunk in enumerate(chunks):
      if len(chunk) > 0:
        text = add_punctuation(chunk)
        text = " ".join(replace_acronym(text.split()))
        print(text.encode('utf-8'))
        wav, _ = synthesizer.synthesize(text, return_wav=True)
        combined_wav = np.concatenate((combined_wav, wav[:-880*6]))

    out = io.BytesIO()
    audio.save_wav(combined_wav, out)
    return out.getvalue()
