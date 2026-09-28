import warnings
from itertools import chain

import numpy as np


def infer_beat_numbers(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """
    From beat and downbeat times, infer a number for each beat such that each downbeat
    is associated with a 1 and beats in between are counted upwards.
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.

    Returns:
        numbers (numpy.ndarray): Array of integer beat numbers.
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

    # handle pickup measure, by considering the beat count of the first full measure
    if len(downbeats) >= 2:
        # find the number of beats between the first two downbeats
        first_downbeat, second_downbeat = np.searchsorted(beats, downbeats[:2])
        beats_in_first_measure = second_downbeat - first_downbeat
        # find the number of beats before the first downbeat
        pickup_beats = first_downbeat
        # derive where to start counting
        if pickup_beats < beats_in_first_measure:
            start_counter = beats_in_first_measure - pickup_beats
        else:
            warnings.warn(
                "There are more beats in the pickup measure than in the first measure. The beat count will start from 2 without trying to estimate the length of the pickup measure.",
                stacklevel=2,
            )
            start_counter = 1
    else:
        warnings.warn(
            "There are less than two downbeats in the predictions. Something may be wrong. The beat count will start from 2 without trying to estimate the length of the pickup measure.",
            stacklevel=2,
        )
        start_counter = 1

    # assemble the beat numbers
    numbers = []
    counter = start_counter
    downbeats = chain(downbeats, [-1])
    next_downbeat = next(downbeats)
    for beat in beats:
        if beat == next_downbeat:
            counter = 1
            next_downbeat = next(downbeats)
        else:
            counter += 1
        numbers.append(counter)
    return np.asarray(numbers)


def replace_state_dict_key(state_dict: dict, old: str, new: str):
    """Replaces `old` in all keys of `state_dict` with `new`."""
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict
