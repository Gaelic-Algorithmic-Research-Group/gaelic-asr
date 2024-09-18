import sys

import librosa
import numpy as np
import onnxruntime as rt

from kaldi.base import set_verbose_level, get_verbose_level
from kaldi.asr import NnetLatticeFasterRecognizer, LatticeRnnlmPrunedRescorer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.lat.functions import ComposeLatticePrunedOptions
from kaldi.matrix import Matrix
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.rnnlm import RnnlmComputeStateComputationOptions


def create_asr(model_path):
    xlsr_model = load_xlsr_model(model_path)
    asr_model = load_asr_model(model_path)
    rnnlm_model, symbols = load_rnnlm(model_path)

    return ASR(xlsr_model, asr_model, rnnlm_model, symbols)

def load_xlsr_model(path):
    sess_opt = rt.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    xlsr_model = rt.InferenceSession(f'{path}/xls_r_300m_cp_18.onnx', sess_opt)

    return xlsr_model

def load_asr_model(path):
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 15
    decoder_opts.lattice_beam = 4
    decoder_opts.max_active = 7000
    decoder_opts.min_active = 200
    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 1

    return NnetLatticeFasterRecognizer.from_files(
        f'{path}/final.mdl', f'{path}/HCLG.fst', f'{path}/words.txt',
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

def load_rnnlm(path):
    symbols = SymbolTable.read_text(f"{path}/words.txt")
    rnnlm_opts = RnnlmComputeStateComputationOptions()
    rnnlm_opts.bos_index = symbols.find_index("<s>")
    rnnlm_opts.eos_index = symbols.find_index("</s>")
    rnnlm_opts.brk_index = symbols.find_index("<brk>")
    compose_opts = ComposeLatticePrunedOptions()
    compose_opts.lattice_compose_beam = 4
    rnnlm = LatticeRnnlmPrunedRescorer.from_files(
        f"{path}/G.fst", f"{path}/rnnlm_word_embedding.raw", f"{path}/rnnlm_final.raw",
        acoustic_scale=1.0, max_ngram_order=4, use_const_arpa=False,
        opts=rnnlm_opts, compose_opts=compose_opts)

    return rnnlm, symbols

def seconds_to_timecode(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}" 

class ASR:

    def __init__(self, xlsr_model, asr_model, rnnlm_model, symbols):
        self.xlsr_model = xlsr_model
        self.asr_model = asr_model
        self.rnnlm_model = rnnlm_model
        self.symbols = symbols

    def transcribe(self, wav, segment_duration=30):
        if len(wav) == 0:
            return ''

        segment_length = int(segment_duration * 16000)
        for start in range(0, wav.shape[0], segment_length):
            segment = wav[start:start + segment_length].astype('float32')
            yield self.decode(self.extract_features(segment))

    def extract_features(self, wav):
        chunks = []
        chunk_duration = 2.5
        chunk_length = int(chunk_duration * 16000)
        for start in range(0, wav.shape[0], chunk_length):
            chunk = wav[start:start + chunk_length].astype('float32')
            chunks.append(self.xlsr_model.run(['output'], {'input': chunk.reshape((1, -1))})[0])

        return np.concatenate(chunks, axis=1)[0]

    def decode(self, feats):
        out = self.asr_model.decode(Matrix(feats))
        rescored_lat = self.rnnlm_model.rescore(out["lattice"])
        words = get_linear_symbol_sequence(shortestpath(rescored_lat))[0]
        output = " ".join(indices_to_symbols(self.symbols, words))
        output = output.replace('@@ @@', '').replace(' dh ', " dh'")
        return output


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: OMP_NUM_THREADS=4 python3 asr.py model wav')

    asr = create_asr(sys.argv[1])
    wav, _ = librosa.load(sys.argv[2], mono=True, sr=16000)
    segment_duration = 10
    for i, segment_transcript in enumerate(asr.transcribe(wav, segment_duration)):
        print(i + 1)
        print(f"{seconds_to_timecode(i * segment_duration)} --> {seconds_to_timecode((i+1) * segment_duration)}")
        print(segment_transcript)
        print()
