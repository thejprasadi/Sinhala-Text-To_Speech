import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import plot

value = input("Please enter a string:\n")
sentences = [
    # From July 8, 2017 New York Times:
    value
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  print(sentences)
  for i, text in enumerate(sentences):
    print(i)
    print(text)
    wav_path = '%s-%d.wav' % (base_path, i)
    align_path = '%s-%d.png' % (base_path, i)
    print('Synthesizing and plotting: %s' % wav_path)
    wav, alignment = synth.synthesize(text)
    with open(wav_path, 'wb') as f:
      f.write(wav)
    plot.plot_alignment(
        alignment, align_path,
        info='%s' % (text)
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint', required=True,
      help='Path to model checkpoint')
  parser.add_argument(
      '--hparams', default='',
      help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument(
      '--force_cpu', default=False,
      help='Force synthesize with cpu')
  parser.add_argument(
      '--gpu_assignment', default='0',
      help='Set the gpu the model should run on')

  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment

  if args.force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
