"""Inference code vendored from beat_this 1.1.0 (https://github.com/CPJKU/beat_this).

MIT License, Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria; see LICENSE.

Modifications against upstream 1.1.0:
- model.py is model/beat_tracker.py and roformer.py is model/roformer.py; imports made relative.
- postprocessor.py: the madmom DBN branch is removed; Postprocessor accepts only type="minimal".
- preprocessing.py: load_audio (torchaudio / soundfile / madmom fallbacks) is removed; LogMelSpect is kept.
- inference.py: CHECKPOINT_URL, load_checkpoint, load_model, File2Beats and File2File are removed, so nothing
  here downloads or unpickles weights. Spect2Frames, Audio2Frames and Audio2Beats take an already loaded model
  instead of a checkpoint path, and the dbn parameter is removed.
- utils.py: only infer_beat_numbers and replace_state_dict_key are kept; the warnings of infer_beat_numbers are
  emitted with warnings.warn instead of print.
- cli.py, model/pl_module.py, model/loss.py, dataset/* and launch_scripts/* are not included.
"""
