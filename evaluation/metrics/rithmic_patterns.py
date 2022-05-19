import numpy as np
from matplotlib import pyplot as plt


def matrix_of_adjacent_rihmic_patterns(roll_or_song, voice='melody'):
    rps = roll_or_song.get_adjacent_rithmic_patterns(voice)

    return np.histogram2d(rps[:-1], rps[1:]) #A chequear


def plot_matrix_of_adjacent_rithmic_patterns(roll_or_song, voice='melody'):
	"""
	We define rithmic patterns for each beat of quarter note. Considering a
	granularity of sixteenth note, and calling:
	c = crotchet (quarter note)
	q = quaver (eighth note)
	d = dotter quaver
	s = semiquaver (sixteenth note)
	r = semiquaver beat of rest
	Thus, we have 16 different patterns
	(c, qq, qss, sqs, ssq, ds, sd, ssss, rrq, qrr, rqr, rrrs, rrsr, rsrr, srrr, rrrr)

	"""
    rps = roll_or_song.get_adjacent_rithmic_patterns(voice)
    ...
