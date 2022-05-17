import numpy as np
from matplotlib import pyplot as plt


def matrix_of_adjacent_rihmic_patterns(roll_or_song, voice='melody'):
    intervals = roll_or_song.get_adjacent_rithmic_patterns(voice)

    return np.histogram2d(intervals[:-1], intervals[1:])


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
    intervals = roll_or_song.get_adjacent_rithmic_patterns(voice)

    p = plt.hist2d(intervals[:-1], intervals[1:])
    plt.show()
    return p
