#!/usr/bin/env python3
""" Sources:
    [Luz2016survey] Luz, Eduardo José da S., et al. "ECG-based heartbeat classification for arrhythmia detection: A survey." Computer methods and programs in biomedicine 127 (2016): 144-164.
    [Li1995detection] Li, Cuiwei, Chongxun Zheng, and Changfeng Tai. "Detection of ECG characteristic points using wavelet transforms." IEEE Transactions on biomedical Engineering 42.1 (1995): 21-28.
"""


if __name__ == "__main__":
    print("COMPUTER SAYS NO. (Don't rund this module directly!)")

import math
import time

import numpy as np # numpy
import pywt # pywavelets: https://pywavelets.readthedocs.io
import wfdb # WFDB: https://wfdb.readthedocs.io

A_4 = 0.5 # TODO this is a magic value, seen empirically
A_3 = A_4 / 2.0
A_2 = A_3 / 2.0
A_1 = A_2 / 2.0

DURATION = 2 # [seconds]

MM_DET_NEIGH_THRESH = 1.20 # Modulus Maximum Detection Neighborhood Threshold for R Peak detection 2) Step 2

NEIGHBORHOOD = 0.020 # [seconds] width for R-Peak detection algorithm 
REDUNDANCY_NEIGHBORHOOD = 0.120 # [seconds] width for R-Peak detection algorithm

# TODO: find good ALPHAAP threshold...
ALPHAAP_THRESHOLD = 0.010 # [power]
INTERVAL_THRESHOLD = 0.120 # [seconds]
PT_THRESHOLD = 0.2 # [],  * epsilon[3], empirically
BLANKING_PERIOD = 0.200 # [seconds]
MAX_BACKTRACK_INTERVAL = 1.50 # [seconds]

## ECG characteristics on healthy people
P_WAVE = 0.110 # [seconds], +/- 0.020, from [Luz2016survey]
PQ_INTERVAL = 0.160 + 0.040 # [seconds], +/- 0.040, from [Luz2016survey]
QRS_WIDTH = 0.100 # [seconds], +/- 0.020, from [Luz2016survey]
QT_INTERVAL = 0.400 + 0.040 # [seconds], +/- 0.040, from [Luz2016survey]


def is_extremum(vals):
    if abs(vals[0]) > abs(vals[1]): # previous value is larger
        return False # not a maximum
    if abs(vals[2]) > abs(vals[1]): # following value is larger
        return False # not a maximum
    return True


def zero_crossing(x1, x2, y1, y2):
    """ calculates the Nullschnittstelle of a given Gerade
        Not needed in ecgpointdetector anymore.
    """
    if x1 == x2:
        print("[DEBUG] zero_crossing x1==x2... ecgpointfinder L51")
        return x1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    result = -(b/a)
    if math.isnan(result):
        print("[ERROR] zero_crossing isnan ecgpointfinder L57")
        result = -99
    if math.isinf(result):
        print("[ERROR] zero_crossing isinf ecgpointfinder L60")
        result = -99
    return result


class ECGPointDetector:
    """ Detects characteristic ECG points. See [Li1995detection]
    """

    def __init__(self, sample_rate, max_bpm=200, wavelet="haar"):
        """ Initializes an ECGPointDetector
        """
        self.sample_rate = sample_rate
        self.wavelet = wavelet
        self.max_bpm = max_bpm
        self.reinit()


    def reinit(self):
        """ Initializes an ECGPointDetector
        """
        self.As = [A_1, A_2, A_3, A_4] # Thresholds, see [Li1995detection]
        self.epsilons = [0.3 * A_1, 0.3 * A_2, 0.3 * A_3, 0.3 * A_4] # Thresholds, see [Li1995detection]
        self.alphaap_threshold = ALPHAAP_THRESHOLD
        self.blanking_samples = int (BLANKING_PERIOD * self.sample_rate)
        self.last_rr_interval = MAX_BACKTRACK_INTERVAL # [seconds]
        self.signal = [0] * (DURATION * self.sample_rate) # list for signal
        self.coeffs = [[0] * (DURATION * self.sample_rate), [0] * (DURATION * self.sample_rate), [0] * (DURATION * self.sample_rate), [0] * (DURATION * self.sample_rate)]
        self.head = len(self.signal) # where does the data begin right now?
        self.window = int((60.0 / self.max_bpm) * self.sample_rate)
        self.last_R_peak = -self.last_rr_interval


    def add_signal(self, signal):
        """ adds given points to a signal and returns (new) characteristic points
        """
        # TODO insert length check of signal here
        points = len(signal)
        swt = pywt.swt(data=signal,
                    wavelet=self.wavelet,
                    level=4,
                    start_level=0,
                    axis=-1)
        levels = len(swt)
        coeffs = [swt[levels-1][1],
                  swt[levels-2][1],
                  swt[levels-3][1],
                  swt[levels-4][1]]
        # shift signal
        self.signal[:-points] = self.signal[points:]
        self.signal[-points:] = signal[:]
        # shift coeffs
        for scale in range(4):
            self.coeffs[scale][:-points] = self.coeffs[scale][points:]
            self.coeffs[scale][-points:] = coeffs[scale][:]
        # set starting point for algorithm
        self.last_R_peak -= points
        self.head = max(0, self.head - points)
        start = max(self.head, min(self.last_R_peak + self.blanking_samples, points - len(signal)))
        # run algorithm
        ps, qs, rs, ss, ts = self.find_characteristic_points(self.coeffs,start_sample=start)
        return ps, qs, rs, ss, ts


    def do_swt_and_find_characteristic_points(self, signal, sample_rate, max_bpm):
        self.reinit()
        swt = pywt.swt(data=signal,
                    wavelet=self.wavelet,
                    level=4,
                    start_level=0,
                    axis=-1)
        levels = len(swt)
        coeffs = [swt[levels-1][1],
                swt[levels-2][1],
                swt[levels-3][1],
                swt[levels-4][1]]
        ps, qs, rs, ss, ts = self.find_characteristic_points(coeffs=coeffs, start_sample=0)
        return ps, qs, rs, ss, ts, coeffs


    def find_characteristic_points(self, coeffs, start_sample=0):
        """ R Peak detection algorithm from
        "Detection of ECG Characteristic Points Using Eavelet Transforms" by
        Cuiwei Li, Chongxun Zheng, and Changfeng Tai (1995)
        Parameters:
            coeffs: swt of the signal to be investigated
        Returns:
            ps, qs, rs, ss, ts
        """
        ## 1) selection of characteristic scales
        ## scale s=2^j, 0 < j < 4 (from paper)
        ## 2) Determination of Modulus Maximum Lines of R Waves:
        ## Modulus Maximum (MM): absolute values of all Maxima / Minima
        ## 2.1) find all the maxima larger than epsilon_4
        num_samples = len(coeffs[0]) # should be equal to len(coeffs[0])
        p_time_window = int(PQ_INTERVAL * self.sample_rate) # distance from P to Q in samples
        t_time_window = int(QT_INTERVAL * self.sample_rate) # distance from Q to T in samples
        start, end = start_sample, min(start_sample + self.window, num_samples) # normally we should get more than (window) samples, but you never know...
        backtracked = False
        last_backtrack_end = 0
        n_ks = [[], [], [], []] # n_k^scale from the paper, i.e. the position of a MM in coeffs[scale]
        r_peaks = [] # detected r_peaks
        qrs_onsets = []
        qrs_offsets = []
        t_waves = [] # [(onset, high, offset), (onset, high, offset), ...]
        p_waves = [] # [(onset, high, offset), (onset, high, offset), ...]
        while start < num_samples: # window loop
            n_ks_window = [[], [], [], []] # n_k^scale from the paper, i.e. the position of a MM in coeffs[scale]. Only for this window!
            r_peaks_window = [] # in this window detected r_peaks
            qrs_onsets_window = []
            qrs_offsets_window = []
            t_waves_window = []
            p_waves_window = []
            found_r_peak = False # we could just use len(r_peaks_found)
            scale = 3 # begin with scale 2^4
            for coeff_index in range(start, end): # find MMs for scale 3
                epsilon = self.epsilons[scale] if not backtracked else self.epsilons[scale] * 0.5
                if abs(coeffs[scale][coeff_index]) > epsilon: # greater than threshold
                    if coeff_index > 0 and coeff_index < num_samples - 1 and \
                        is_extremum(coeffs[scale][coeff_index-1:coeff_index+2]):
                        n_ks_window[scale].append(coeff_index)
            ## Now n_ks_window[3] is a list of all (local) extrema > threshold epsilons[3]
            ## 2.2) look at each position if there is a neighborhood maximum at j-1
            neighborhood = int(NEIGHBORHOOD * self.sample_rate) # neighborhood in samples
            for scale in range(3, 0, -1): # find corresponding MMs for lower scales
                n_k_index, goal = 0, len(n_ks_window[scale]) # loop conditions
                while n_k_index < goal: # for every modulus maximum in scale
                    n_k = n_ks_window[scale][n_k_index]
                    if n_k == 0: # DEBUG: should never end up here
                        print("[ERROR]", "This should not happen...")
                        n_ks_window[scale-1].append(0)
                        continue # skip this n_k, as it was invalidated earlier
                    locations = [] # locations of neighboring MMs in lower scale
                    epsilon = self.epsilons[scale] if not backtracked else self.epsilons[scale] * 0.5
                    ## first add modulus maximum at same position (if MM)
                    if abs(coeffs[scale-1][n_k]) > epsilon: # above threshold
                        if is_extremum(coeffs[scale-1][n_k-1:n_k+2]):
                            if (coeffs[scale-1][n_k] > 0) == (coeffs[scale][n_k] > 0): # same signum - this step is *NOT* described in the paper!
                                locations.append(n_k)
                    ## then add MMs of the neighborhood, with increasing distance
                    for i in range(1, neighborhood): # for "the neighborhood"
                        pos = n_k - i
                        if pos > 0:
                            if abs(coeffs[scale-1][pos]) > epsilon: # above threshold
                                if is_extremum(coeffs[scale-1][pos-1:pos+2]):
                                    if (coeffs[scale-1][n_k-i] > 0) == (coeffs[scale][n_k] > 0): # same signum - this step is *NOT* described in the paper!
                                        locations.append(pos)
                        pos = n_k + i
                        if pos < num_samples - 1:
                            if abs(coeffs[scale-1][pos]) > epsilon: # above threshold
                                if is_extremum(coeffs[scale-1][pos-1:pos+2]):
                                    if (coeffs[scale-1][n_k+i] > 0) == (coeffs[scale][n_k] > 0): # same signum - this step is *NOT* described in the paper!
                                        locations.append(pos)
                    toappend = 0
                    if len(locations) == 0: # no modulus maxima on lower scale were found
                        for i in range(scale, 4): # delete all higher scale n_ks_window
                            del n_ks_window[i][n_k_index]
                        goal -= 1
                        continue
                    elif len(locations) == 1: # exactly one modulus maximum
                        toappend = locations[0]
                    else: # more than one modulus maximum
                        ## select largest one
                        vals = []
                        for location in locations:
                            vals.append(abs(coeffs[scale-1][location]))
                        maxindex = vals.index(max(vals))
                        ## if largest one !> MM_DET_NEIGH_THRESH others: select nearest one (MM_DET_NEIGH_THRESH = 1.2 in paper)
                        for val in vals:
                            if val != vals[maxindex]:
                                if vals[maxindex] <= MM_DET_NEIGH_THRESH * val:
                                    ## select nearest value, conveniently first one in the array
                                    maxindex = 0
                                    break
                        toappend = locations[maxindex]
                    if toappend in n_ks_window[scale-1]: # value already in list
                        for i in range(scale, 4): # delete all higher scale n_ks_window
                            del n_ks_window[i][n_k_index]
                        goal -= 1
                        continue
                    else: # append value to n_ks_window
                        n_ks_window[scale-1].append(toappend)
                    n_k_index += 1
            ## eliminate MM lines where n_ks_window[0] == 0, i.e. no MM line at scale 0
            i, goal = 0, len(n_ks_window[0]) # loop conditions
            while i < goal: # clean MM lines where scale 0 is not a MM
                if n_ks_window[0][i] == 0:
                    print("[ERROR] This should not happen")
                    for scale in range(4):
                        del n_ks_window[scale][i]
                    goal -= 1
                    continue
                i += 1
            ## 3) Calculation of Singular Degree
            def a(j, n_k): # see paper
                result = abs(coeffs[j][n_k])
                if result == 0:
                    result = 0.00001
                return result
            def alpha(j, n_k): # see paper
                try:
                    return math.log2(a(j+1, n_k)) - math.log2(a(j, n_k))
                except ValueError as ve:
                    print("[ERROR] exception...", ve)
                    print("[DEBUG] a(j+1, n_k)", a(j+1, n_k), "a(j, n_k)", a(j, n_k))
                return 0
            def alphaap(n_k): # alphaap = alpha apostrohpe = \alpha', see paper
                return (alpha(0, n_k) + alpha(1, n_k)) / 2
            ## eliminate lines where alphaap < threshold.
            ## quote from the paper:
            ## "(...) if the \alpha' suddenly decreases or even becomes negative, the
            ##  corresponding singularity point (...) will be eliminated"
            i, goal = 0, len(n_ks_window[0]) # loop conditions
            while i < goal: # delete MM line of all noise singularities
                n_k = n_ks_window[0][i]
                if alphaap(n_k) < self.alphaap_threshold: # Singularity probably is noise
                    for scale in range(4): # delete MM line on all scales
                        del n_ks_window[scale][i]
                    goal -= 1
                    continue
                self.alphaap_threshold = 0.1 * alphaap(n_k) # "if the \alpha' suddenly decreases greatly (...) the corresponding singularity point must be noise"
                i += 1
            ## 4) Elimination of Isolation and Redundant Modulus Maximum Lines:
            ## 4.1) "First, eliminiate isolation modulus maximum lines."
            n_k_index, goal = 0, len(n_ks_window[0]) # loop conditions
            while n_k_index < goal: # delete isolation MM lines
                n_k = n_ks_window[0][n_k_index]
                signum = (coeffs[0][n_k] > 0) # signum of this n_k
                ## find previous MM with different signum
                previous_n_k_index = n_k_index - 1
                previous_n_k = 0
                while previous_n_k_index >= 0: # find previous MM with different signum
                    previous_n_k = n_ks_window[0][previous_n_k_index]
                    if (coeffs[0][previous_n_k] > 0) != signum: # different signums
                        break
                    previous_n_k_index -= 1 # continue loop
                if previous_n_k_index >= 0: # compare with previous value
                    if n_k - previous_n_k < int(INTERVAL_THRESHOLD * self.sample_rate): # not isolation line
                        n_k_index += 1
                        continue
                ## find next MM with different signum
                next_n_k_index = n_k_index + 1
                next_n_k = 0
                while next_n_k_index < goal: # find next MM with different signum
                    next_n_k = n_ks_window[0][next_n_k_index]
                    if (coeffs[0][next_n_k] > 0) != signum: # different signums
                        break
                    next_n_k_index += 1 # continue loop
                if next_n_k_index < goal: # compare with next value
                    if next_n_k - n_k < int(INTERVAL_THRESHOLD * self.sample_rate): # not isolation line
                        n_k_index += 1
                        continue
                ## if we are here, this is an isolation line
                for scale_index in range(4): # eliminate isolation MM line on all scales
                    del n_ks_window[scale_index][n_k_index]
                    goal -= 1
            ## 4.2) "Next, eliminate redundant modulus maximum lines."
            n_k_index, goal = 0, len(n_ks_window[2]) # loop conditions
            while n_k_index < goal: # remove redundant MM lines
                n_k = n_ks_window[2][n_k_index]
                signum = (coeffs[2][n_k] > 0)
                neighborhood_mms = [] # (n_k, s_j(n), dist to n_k)
                ## gather earlier MMs in neighborhood
                previous_n_k_index = n_k_index - 1 # loop condition
                while previous_n_k_index >= 0: # gather previous MMs in neighborhood
                    previous_n_k  = n_ks_window[2][previous_n_k_index]
                    if n_k - previous_n_k < int(REDUNDANCY_NEIGHBORHOOD * self.sample_rate): # previous_n_k is in the neighborhood of n_k
                        if (coeffs[2][previous_n_k] > 0) != signum: # other signum than n_k
                            neighborhood_mms.append((previous_n_k, coeffs[2][previous_n_k], n_k - previous_n_k))
                        previous_n_k_index -= 1 # loop continuation
                    else: # distance > neighborhood -> break
                        break
                ## gather later MMs
                next_n_k_index = n_k_index + 1
                while next_n_k_index < goal: # gather later MMs in neighborhood
                    next_n_k = n_ks_window[2][next_n_k_index]
                    if next_n_k - n_k < int(REDUNDANCY_NEIGHBORHOOD * self.sample_rate):
                        if (coeffs[2][next_n_k] > 0) != signum: # other signum than n_k
                            neighborhood_mms.append((next_n_k, coeffs[2][next_n_k], next_n_k - n_k))
                        next_n_k_index += 1 # loop continuation
                    else:
                        break # distance > neighborhood -> break
                if len(neighborhood_mms) < 2: # only one neighbor --> no redundant neighbors to eliminate --> continue
                    n_k_index += 1 # loop continuation
                    continue
                num_neighbors = len(neighborhood_mms) # loop condition
                while num_neighbors > 1: # pairwise compare and eliminate neighbors
                    try:
                        neighbor1 = neighborhood_mms[0]
                        neighbor2 = neighborhood_mms[1]
                        delindex = 0
                        ## "Rule 1: If A_1/L_1 > 1.2 A_2/L_2: Min2 is redundant."
                        if abs(neighbor1[1]) / abs(neighbor1[2]) > (abs(neighbor2[1]) / abs(neighbor2[2])) * 1.20:
                            delindex = n_ks_window[2].index(neighbor2[0])
                            del neighborhood_mms[1]
                        ## "Rule 2: If A_2/L_2 > 1.2 A_1/L_1: Min1 is redundant."
                        elif abs(neighbor2[1]) / abs(neighbor2[2]) > (abs(neighbor1[1]) / abs(neighbor1[2])) * 1.20:
                            if neighbor1[0] in n_ks_window[2]:
                                delindex = n_ks_window[2].index(neighbor1[0])
                            else:
                                print("[ERROR] [DEBUG] damn 307", neighbor1[0], n_ks_window[2])
                            del neighborhood_mms[0]
                        else: # "Rule 3: Otherwise, "
                            if (neighbor1[0] < n_k) == (neighbor2[0] < n_k): # "both are on the same side of the positive maximum"
                                delindex = n_ks_window[2].index(neighbor2[0]) # "the minimum farther from the maximum is redundant"
                                del neighborhood_mms[1]
                            else: # "Min1 and Min2 are on different sides of the maximum"
                                if coeffs[2][n_k] > 0: # n_k is positive maximum
                                    delindex = n_ks_window[2].index(neighbor2[0]) # "the minimum following the maximum is redundant"
                                else: # n_k is negative minimum
                                    if neighbor1[0] in n_ks_window[2]:
                                        delindex = n_ks_window[2].index(neighbor1[0]) # the maximum before the minimum is redundant
                                    else:
                                        print("[ERROR] [DEBUG] line 320", neighbor1[0], n_ks_window[2])
                                del neighborhood_mms[1]
                        for scale in range(4): # eliminate redundant MM line
                            del n_ks_window[scale][delindex]
                        num_neighbors -= 1 # inner loop stop condition
                        goal -= 1 # outer loop stop condition
                    except ValueError as ve:
                        print("[ERROR] [DEBUG] damn 327", ve)
            ## 5) Detection of the R Peak:
            ## "R peak can be located at a zero-crossing point of a positive maximum-negative minimum pair at scale 2^1."
            n_k_index, goal = 0, len(n_ks_window[0])
            while n_k_index < goal - 1: # find MM pairs
                x1 = n_ks_window[0][n_k_index]
                x2 = n_ks_window[0][n_k_index + 1]
                ## TODO / DEBUG this is not needed...
                # find *real* minimum / maximum on coeffs:
                # x1start = max(0, x1-2*neighborhood)
                # x1end = min(num_samples, x1+2*neighborhood)
                # if coeffs[0][x1] < 0:
                #     x1 = x1start + np.argmin(coeffs[0][x1start:x1end]).item()
                #     n_ks_window[0][n_k_index] = x1
                # else:
                #     x1 = x1start + np.argmax(coeffs[0][x1start:x1end]).item()
                #     n_ks_window[0][n_k_index] = x1
                # x2start = max(0, x2-2*neighborhood)
                # x2end = min(num_samples, x2+2*neighborhood)
                # if coeffs[0][x2] < 0:
                #     x2 = x2start + np.argmin(coeffs[0][x2start:x2end]).item()
                #     n_ks_window[0][n_k_index+1] = x2
                # else:
                #     x2 = x2start + np.argmax(coeffs[0][x2start:x2end]).item()
                #     n_ks_window[0][n_k_index+1] = x2
                y1 = coeffs[0][x1]
                y2 = coeffs[0][x2]
                # zero_point = int(zero_crossing(x1, x2, y1, y2)) + 2 # there seems to be some drift, thats why + 2
                zero_point = int((x1+x2)/2)
                if zero_point > 0 and zero_point < num_samples: # the values should be *inside* the numbers
                    if zero_point - self.last_R_peak > self.blanking_samples: # blanking
                        r_peaks_window.append(zero_point)
                        found_r_peak = True
                        self.last_R_peak = r_peaks_window[-1] # last R Peak
                        for scale in range(4): # update epsilon thresholds
                            max_abs = max(abs(coeffs[scale][n_ks_window[scale][-2]]), abs(coeffs[scale][n_ks_window[scale][-1]]))
                            if max_abs < 2.0 * self.As[scale]:
                                #A_before = As[scale]
                                self.As[scale] = 0.875 * self.As[scale] + 0.125 * max_abs
                                self.epsilons[scale] = 0.3 * self.As[scale]
                        ## QRS onset and offset:
                        ## For every R peak: Look for MM before and after the MM Line, track it to zero and this should be the On- and Offset
                        found_q, found_s = False, False
                        ## look for MM before x1
                        while x1 > 0:
                            if (coeffs[1][x1] > 0) != (y1 > 0):
                                break
                            x1 -= 1
                        found_q = True
                        qrs_onsets_window.append(x1)
                        while x2 < num_samples:
                            if (coeffs[1][x2] > 0) != (y2 > 0):
                                break
                            x2 += 1
                        found_q = True
                        qrs_offsets_window.append(x2)
                        # for direction in [-1, +1]:
                        #     distance = 0
                        #     modmaxfound = False
                        #     x, y = x1, y1
                        #     if direction == +1:
                        #         x, y = x2, y2
                            # while abs(distance) < 4*neighborhood:
                            #     if modmaxfound and ((coeffs[0][x + distance] >= 0) == (y >= 0)): # same signum
                            #         if direction == -1:
                            #             found_q = True
                            #             qrs_onsets_window.append(x + distance)
                            #         else:
                            #             found_s = True
                            #             qrs_offsets_window.append(x + distance)
                            #         break
                            #     elif ((coeffs[0][x + distance] >= 0) != (y >= 0)) and is_extremum(coeffs[0][x+distance-1:x+distance+2]):# and (abs(coeffs[0][x + distance]) > 0.5 * epsilons[0]): # other signum
                            #         modmaxfound = True
                            #     distance += direction
                            #     if x + distance < 1 or x+ distance >= num_samples-2:
                            #         break
                        ## T wave detection
                        scale = 3 # from [Luz2016survey]
                        offset = 7 * neighborhood # TODO make this a variable? Work with QT Interval?
                        t_start = max(0, self.last_R_peak + offset)
                        t_end = min(num_samples, self.last_R_peak + t_time_window)
                        if found_s:
                            t_start = max(0, qrs_offsets_window[-1] + offset)
                            t_end = min(num_samples, qrs_offsets_window[-1] + t_time_window)
                        if t_start < t_end:
                            negmin = t_start + np.argmin(coeffs[scale][t_start:t_end]).item()
                            posmax = t_start + np.argmax(coeffs[scale][t_start:t_end]).item()
                            x1, x2 = min(negmin, posmax), max(negmin, posmax)
                            y1 = coeffs[scale][x1]
                            y2 = coeffs[scale][x2]
                            threshold = PT_THRESHOLD * self.epsilons[scale] # Thresholding
                            if (abs(y1) > threshold) and (abs(y2) > threshold):
                                # zero_point = int(zero_crossing(x1, x2, y1, y2))
                                zero_point = int((x1 + x2) / 2)
                                while x1 > 0: # von x1 nach links zum nächsten MM oder 0
                                    if (coeffs[scale][x1] > 0) != (y1 > 0):
                                        break
                                    x1 -= 1
                                while x2 < num_samples: # von x1 nach links zum nächsten MM oder 0
                                    if (coeffs[scale][x2] > 0) != (y2 > 0):
                                        break
                                    x2 += 1
                                t_waves_window.append((x1, zero_point, x2))
                        ## P wave detection
                        scale = 3 # from [Luz2016survey]
                        offset = 1 * neighborhood # TODO make this a variable?
                        p_start = max(0, self.last_R_peak - p_time_window)
                        p_end = min(num_samples, self.last_R_peak - offset)
                        if found_q:
                            p_start = max(0, qrs_onsets_window[-1] - p_time_window)
                            p_end = min(num_samples, qrs_onsets_window[-1] - offset)
                        if p_start < p_end:
                            negmin = p_start + np.argmin(coeffs[scale][p_start:p_end]).item()
                            posmax = p_start + np.argmax(coeffs[scale][p_start:p_end]).item()
                            x1, x2 = min(negmin, posmax), max(negmin, posmax)
                            y1 = coeffs[scale][x1]
                            y2 = coeffs[scale][x2]
                            threshold = PT_THRESHOLD * self.epsilons[scale] # Thresholding
                            if (abs(y1) > threshold) and (abs(y2) > threshold):
                                # zero_point = int(zero_crossing(x1, x2, y1, y2))
                                zero_point = int((x1 + x2) / 2)
                                while x1 > 0: # von x1 nach links zum nächsten MM oder 0
                                    if (coeffs[scale][x1] > 0) != (y1 > 0):
                                        break
                                    x1 -= 1
                                while x2 < num_samples: # von x1 nach links zum nächsten MM oder 0
                                    if (coeffs[scale][x2] > 0) != (y2 > 0):
                                        break
                                    x2 += 1
                                p_waves_window.append((x1, zero_point, x2))
                    else:
                        for scale in range(4): # delete blanked n_ks
                            del n_ks_window[scale][n_k_index] # yes, twice, we need to delete two values
                            del n_ks_window[scale][n_k_index] # yes, twice, we need to delete two values
                        goal -= 2
                        continue
                n_k_index += 2 # loop continuation
            ## back in window loop
            for i in range(4): # merge n_ks_window
                for n_k in n_ks_window[i]:
                    if not n_k in n_ks[i]:
                        n_ks[i].append(n_k)
            for peak in r_peaks_window: # merge R peaks
                if not peak in r_peaks:
                    r_peaks.append(peak)
            for onset in qrs_onsets_window:
                if not onset in qrs_onsets:
                    qrs_onsets.append(onset)
            for offset in qrs_offsets_window:
                if not offset in qrs_offsets:
                    qrs_offsets.append(offset)
            for p in p_waves_window: # merge P waves
                if not p in p_waves:
                    p_waves.append(p)
            for t in t_waves_window: # merge T waves
                if not t in t_waves:
                    t_waves.append(t)
            if found_r_peak: # found a new r_peak
                start += self.window # window loop continuation
                if len(r_peaks) > 1:
                    self.window = r_peaks[-1] - r_peaks[-2] # last RR interval
                    self.last_rr_interval = (r_peaks[-1] - r_peaks[-2]) / self.sample_rate
                else:
                    self.window = max(self.window, r_peaks[-1]) # We have no idea about the last RR interval, so just take the window or distance from beginning of the signal
                backtracked = False
            else: # found no new R Peak...
                if backtracked: # we already backtracked... there does *really* not seem to be a R peak in this window...
                    backtracked = False
                    last_backtrack_end = end
                    start += self.window
                else: # we have not yet backtracked
                    if len(r_peaks) > 0: # if we have found any R peaks already
                        interval = self.last_rr_interval * self.sample_rate# in seconds
                        if end - r_peaks[-1] > int(1.5 * interval):
                            backtracked = True
                            # TODO probably the paper means something else in "backtracking", i.e. only scale3 comparision?
                            start = max(last_backtrack_end, r_peaks[-1] + self.blanking_samples)
                        else:
                            start += int(0.5 * self.window)
                    else: # we have not found a single R peak yet...
                        start += self.window
            end = min(start + self.window, num_samples) # window loop continuation
        ## 6) (not in paper): find peak in signal
        ## TODO DEBUG THIS IS NOT NEEDED
        # for peak_index in range(len(r_peaks)):
        #     i = r_peaks[peak_index]
        #     while abs(signal[i-1]) >= abs(signal[i]):
        #         i -= 1
        #     while abs(signal[i+1]) >= abs(signal[i]):
        #         i += 1
        #     r_peaks[peak_index] = i
        return p_waves, qrs_onsets, r_peaks, qrs_offsets, t_waves

