import argparse
import tensorflow as tf
from synthesizer import Synthesizer
from models import create_model
from hparams import hparams, hparams_debug_string
from util import audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path', required=True, help='path to model checkpoint'
    )
    parser.add_argument(
        '--export_path', required=True, help='path to export model'

    )
    args = parser.parse_args()

    builder = tf.saved_model.builder.SavedModelBuilder(args.export_path)

    synth = Synthesizer()
    synth.load(args.checkpoint_path)

    inputs = tf.saved_model.utils.build_tensor_info(synth.model.inputs)
    input_lengths = tf.saved_model.utils.build_tensor_info(
        synth.model.input_lengths
    )

    w_o = audio.inv_spectrogram_tensorflow(
        synth.model.linear_outputs
    )

    wav_output = tf.saved_model.utils.build_tensor_info(w_o)
    alignment = tf.saved_model.utils.build_tensor_info(synth.model.alignments)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'inputs': inputs,
                "input_lengths": input_lengths
            },
            outputs={
                'wav_output': wav_output,
                'alignment': alignment
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

    with synth.session as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict': prediction_signature}
        )
        builder.save()
    print("exported .pb to", args.export_path)
