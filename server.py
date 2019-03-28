#!/usr/bin/env python3
"""
WS server that reads data from the ADS1293 and sends it periodically to all connected websockets
"""

import asyncio
import json
import logging
import numpy as np
import pywt # pywavelets: https://pywavelets.readthedocs.io
import threading
import time
import websockets

import ADS1293 # the ADS1293 module does all the fancy hardware magic
from ecgpointdetector import ECGPointDetector # the ecg-point-finder.py file contains the magic algorithm

HOST = ''
PORT = 9669

DIAGNOSE_DURATION = 6 # [seconds], must be int!
BATCH_SIZE = 128 # must be divisable by 2^5 at least

LOOP = asyncio.get_event_loop()

STATE = {
    'running': False,
    'diagnosing': False,
    'diagnosestart': 0,
    'clients': 0,
    'ina1max': 0,
    'ina1min': 0,
    'ina1zero': 0,
    'ina2max': 0,
    'ina2min': 0,
    'ina2zero': 0,
    'ina3max': 0,
    'ina3min': 0,
    'ina3zero': 0,
    'leads': 3,
}

CLIENTS = set()
DATA_GENERATOR_THREAD = {'thread': None}

## ECG characteristics on healthy people
P_WAVE_LEN = 0.110 # [seconds], from [Luz2016survey]
P_WAVE_LEN_VAR = 0.020 # [seconds]
PQ_INTERVAL = 0.160 # [seconds], from [Luz2016survey]
PQ_INTERVAL_VAR = 0.040 # [seconds]
QRS_WIDTH = 0.100 # [seconds], from [Luz2016survey]
QRS_WIDTH_VAR = 0.020 # [seconds]
QT_INTERVAL = 0.400 # [seconds],  from [Luz2016survey]
QT_INTERVAL_VAR = 0.040 # [seconds]
P_AMPLITUDE = 0.115 # [mV]
P_AMPLITUDE_VAR = 0.05 # [mV]
QRS_AMPLITUDE = 1.5 # [mV]
QRS_AMPLITUDE_VAR = 0.5 # [mV]
ST_LEVEL = 0.0 # [mV]
ST_LEVEL_VAR = 0.1 # [mV]
T_AMPLITUDE = 0.3 # [mV]
T_AMPLITUDE_VAR = 0.2 # [mV]

#SIGNALS = [[0]*DIAGNOSE_DURATION*ADS1293.SAMPLE_RATE, [0]*DIAGNOSE_DURATION*ADS1293.SAMPLE_RATE, [0]*DIAGNOSE_DURATION*ADS1293.SAMPLE_RATE] # for diagnose mode
SIGNALS = [[], [], []] # for diagnose mode

ECGDETECTOR = ECGPointDetector(sample_rate=ADS1293.SAMPLE_RATE)

SIGNAL = [] # ring buffer for Lead II signal
SIGNAL_TRANSFORM = [] # ring buffer for Lead II signal transform
TAIL = 0 # tail of the ring buffer

def generate_data(leads=3):
    logging.debug("generate_data() started")
    ina1zero = int(STATE['ina1zero'])
    ina2zero = int(STATE['ina2zero'])
    ina3zero = int(STATE['ina3zero'])
    ina1mult = 400.0 / (STATE['ina1max']-STATE['ina1min']) # test signal is 200 milliVolts
    ina2mult = 400.0 / (STATE['ina2max']-STATE['ina2min']) # test signal is 200 milliVolts
    ina3mult = 400.0 / (STATE['ina3max']-STATE['ina3min']) # test signal is 200 milliVolts
    ads = ADS1293.ADS1293()
    if leads == 3:
        ads.init_three_lead_ecg()
    elif leads == 5:
        ads.init_five_lead_ecg()
    i = 0
    data1buffer = [0.0] * BATCH_SIZE
    data2buffer = [0.0] * BATCH_SIZE
    data3buffer = [0.0] * BATCH_SIZE

    thread = threading.currentThread()
    future = None
    while getattr(thread, "do_run", True):
        starttime = time.time()
        if ads.check_ready(): # busy wait condition
            if leads == 3:
                data = ads.read(ADS1293.DATA_LOOP, ADS1293.NUM_ECG_DATA_REGISTERS * 2)
                ecg1 = data[0] << 16 | data[1] << 8 | data[2]
                ecg2 = data[3] << 16 | data[4] << 8 | data[5]
                data1buffer[i % BATCH_SIZE] = (ecg1 - ina1zero) * ina1mult
                data2buffer[i % BATCH_SIZE] = (ecg2 - ina2zero) * ina2mult
                data3buffer[i % BATCH_SIZE] = ecg2 - ecg1
            elif leads == 5:
                data = ads.read(ADS1293.DATA_LOOP, ADS1293.NUM_ECG_DATA_REGISTERS * 3)
                ecg1 = data[0] << 16 | data[1] << 8 | data[2]
                ecg2 = data[3] << 16 | data[4] << 8 | data[5]
                ecg3 = data[6] << 16 | data[7] << 8 | data[8]
                data1buffer[i % BATCH_SIZE] = (ecg1 - ina1zero) * ina1mult
                data2buffer[i % BATCH_SIZE] = (ecg2 - ina2zero) * ina2mult
                data3buffer[i % BATCH_SIZE] = (ecg3 - ina3zero) * ina3mult
            if i % BATCH_SIZE == BATCH_SIZE-1:
                data1 = data1buffer.copy()
                data2 = data2buffer.copy()
                data3 = data3buffer.copy()
                if future:
                    if future.done():
                        future.result()
                        coro = send_data(data1, data2, data3)
                        future = asyncio.run_coroutine_threadsafe(coro, LOOP)
                    else:
                        logging.info("not yet done with old loop, throwing away data...")
                else:
                    coro = send_data(data1, data2, data3)
                    future = asyncio.run_coroutine_threadsafe(coro, LOOP)
            i = i + 1
            timetosleep = (1.0 / ADS1293.SAMPLE_RATE - (time.time()-starttime))/2
            if timetosleep > 0:
                time.sleep(timetosleep)
    # do_run is now false
    ads.reset()
    ads.close()
    logging.debug("generate_data() ended")

async def send_data(data1, data2, data3):
    # global TAIL
    # global SIGNAL
    # global SIGNAL_TRANSFORM
    # swt = pywt.swt(data=data2,
    #                 wavelet="haar",
    #                 level=4,
    #                 start_level=0,
    #                 axis=-1)
    # levels = len(swt)
    # coeffs = [swt[levels-1][1],
    #         swt[levels-2][1],
    #         swt[levels-3][1],
    #         swt[levels-4][1]]
    # for i in range(len(data2)):
    #     SIGNAL[TAIL] = data2[i]
    #     for scale in range(4):
    #         SIGNAL_TRANSFORM[scale][TAIL] = coeffs[scale][i]
    #     TAIL = (TAIL + 1) % (DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE)
    #ps, qs, rs, ss, ts = # [],[],[],[],[] #ECGDETECTOR.add_signal(data2)
    ps, qs, rs, ss, ts = ECGDETECTOR.add_signal(data2)

    if CLIENTS: # asyncio.wait doesn't accept an empty list
        message = json.dumps({
            'type': 'data',
            'data1': data1, # lead I
            'data2': data2, # lead II
            'data3': data3, # lead III or V
            'ps': ps,
            'qs': qs,
            'rs': rs,
            'ss': ss,
            'ts': ts,
        })
        await asyncio.wait([client.send(message) for client in CLIENTS])
    if STATE['diagnosing']:
        SIGNALS[0] = SIGNALS[0] + data1
        SIGNALS[1] = SIGNALS[1] + data2
        SIGNALS[2] = SIGNALS[2] + data3
        now = time.time()
        if now - STATE['diagnosestart'] >= DIAGNOSE_DURATION:
            STATE['diagnosing'] = False
            p_waves, onsets, r_peaks, offsets, t_waves, coeffs = ECGDETECTOR.do_swt_and_find_characteristic_points(signal=SIGNALS[1], sample_rate=ADS1293.SAMPLE_RATE, max_bpm=200)
            ## calculate base line voltage
            base_line_voltage = np.median(SIGNALS[1])
            ## calculate heart rate
            distancesum = 0
            for i in range(len(r_peaks)-1):
                distancesum += r_peaks[i+1] - r_peaks[i]
            avgdist = distancesum / (len(r_peaks)-1)
            diagnose = ""
            if avgdist == 0:
                diagnose = "could not determine heart rate..."
                heart_rate = 999.0
            else:
                heart_rate = 60.0 / (avgdist / ADS1293.SAMPLE_RATE)
                diagnose = "heart rate: %d\n" % int(heart_rate)
            ## check for regular ECG data
            for r_index in range(len(r_peaks)):
                base_line_voltage = np.median(SIGNALS[1][int(r_peaks[r_index] - avgdist/2):int(r_peaks[r_index] + avgdist/2)])
                r_amplitude = SIGNALS[1][r_peaks[r_index]] - base_line_voltage
                if r_amplitude < QRS_AMPLITUDE - QRS_AMPLITUDE_VAR:
                    diagnose += ", QRS Complex %d is low (%fmV)" % (r_index, r_amplitude)
                elif r_amplitude > QRS_AMPLITUDE + QRS_AMPLITUDE_VAR:
                    diagnose += ", QRS Complex %d is high(%fmV)" % (r_index, r_amplitude)
                ## has p wave?
                has_p_wave = False
                p_index = 0
                while p_index < len(p_waves):
                    if r_peaks[r_index] - p_waves[p_index][1] < (PQ_INTERVAL + PQ_INTERVAL_VAR) * ADS1293.SAMPLE_RATE:
                        has_p_wave = True
                        break
                    p_index += 1
                if has_p_wave:
                    p_wave_len = (p_waves[p_index][2] - p_waves[p_index][0]) / ADS1293.SAMPLE_RATE
                    if p_wave_len < (P_WAVE_LEN - P_WAVE_LEN_VAR):
                        diagnose += ", P wave %d is narrow (%fs)" % (r_index, p_wave_len)
                    elif p_wave_len > (P_WAVE_LEN + P_WAVE_LEN_VAR):
                        diagnose += ", P wave %d is wide (%fs)" % (r_index, p_wave_len)
                    ## amplitude of P
                    p_amplitude = SIGNALS[1][p_waves[p_index][1]] - base_line_voltage
                    if p_amplitude < P_AMPLITUDE - P_AMPLITUDE_VAR:
                        diagnose += ", P wave %d is low (%fmV)" % (r_index, p_amplitude)
                    elif p_amplitude > P_AMPLITUDE + P_AMPLITUDE_VAR:
                        diagnose += ", P wave %d is high (%fmV)" % (r_index, p_amplitude)
                else:
                    diagnose += ", P wave %d is missing" % r_index
                ## has t wave?
                has_t_wave = False
                t_index = 0
                while t_index < len(t_waves):
                    if t_waves[t_index][1] - r_peaks[r_index] < (QT_INTERVAL + QT_INTERVAL_VAR) * ADS1293.SAMPLE_RATE:
                        has_t_wave = True
                        break
                    t_index += 1
                if has_t_wave:
                    ## amplitude of T
                    t_amplitude = SIGNALS[1][t_waves[t_index][1]] - base_line_voltage
                    if t_amplitude < T_AMPLITUDE - T_AMPLITUDE_VAR:
                        diagnose += ", T wave %d is low (%fmV)" % (r_index, t_amplitude)
                    elif t_amplitude > T_AMPLITUDE + T_AMPLITUDE_VAR:
                        diagnose += ", T wave %d is high (%fmV)" % (r_index, t_amplitude)
                else:
                    diagnose += ", T wave %d is missing" % r_index
            if CLIENTS:
                message = json.dumps({
                    'type':'diagnose',
                    'data1':SIGNALS[0],
                    'data2':SIGNALS[1],
                    'data3':SIGNALS[2],
                    'coeffs': [[i for i in coeffs[0]], [i for i in coeffs[1]], [i for i in coeffs[2]], [i for i in coeffs[3]]],
                    'r_peaks':r_peaks,
                    'onsets':onsets,
                    'offsets':offsets,
                    't_waves':t_waves,
                    'p_waves':p_waves,
                    'heart_rate':heart_rate,
                    'diagnose':diagnose,
                    'signal_mean':np.mean(SIGNALS[1]),
                    'signal_median':np.median(SIGNALS[1]),
                    'base_line_voltage':base_line_voltage,
                })
                await asyncio.wait([client.send(message) for client in CLIENTS])

def state_event():
    return json.dumps({'type': 'status', **STATE})

async def notify_state():
    if CLIENTS:       # asyncio.wait doesn't accept an empty list
        message = state_event()
        await asyncio.wait([client.send(message) for client in CLIENTS])

async def register(websocket):
    STATE['clients'] = STATE['clients'] + 1
    CLIENTS.add(websocket)

async def unregister(websocket):
    STATE['clients'] = STATE['clients'] - 1
    CLIENTS.remove(websocket)

async def listen(websocket, path):
    global TAIL
    global SIGNAL
    global SIGNAL_TRANSFORM
    await register(websocket)
    try:
        await websocket.send(state_event())
        async for message in websocket:
            data = json.loads(message)
            if data['action'] == 'stop':
                logging.info("got action stop message")
                if DATA_GENERATOR_THREAD['thread']:
                    DATA_GENERATOR_THREAD['thread'].do_run = False # send stop signal to thread
                    DATA_GENERATOR_THREAD['thread'].join() # wait for thread to stop
                    DATA_GENERATOR_THREAD['thread'] = None
                    STATE['running'] = False
                    logging.debug("thread joined")
                else:
                    logging.debug("there is no thread to stop")
                await notify_state()
            elif data['action'] == 'startthree':
                logging.info("got action startthree message")
                if DATA_GENERATOR_THREAD['thread']:
                    logging.debug("not starting thread as one thread already exists")
                else:
                    SIGNAL = [0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE
                    SIGNAL_TRANSFORM = [[0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE] * 4
                    DATA_GENERATOR_THREAD['thread'] = threading.Thread(target=generate_data, args=([3]))
                    DATA_GENERATOR_THREAD['thread'].start()
                    STATE['running'] = True
                    STATE['leads'] = 3
                    logging.debug("started thread")
                await notify_state()
            elif data['action'] == 'startfive':
                logging.info("got action startfive message")
                if DATA_GENERATOR_THREAD['thread']:
                    logging.debug("not starting thread as one thread already exists")
                else:
                    SIGNAL = [0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE
                    SIGNAL_TRANSFORM = [[0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE] * 4
                    DATA_GENERATOR_THREAD['thread'] = threading.Thread(target=generate_data, args=([5]))
                    DATA_GENERATOR_THREAD['thread'].start()
                    STATE['running'] = True
                    STATE['leads'] = 5
                    logging.debug("started thread")
                await notify_state()
            elif data['action'] == 'status':
                await websocket.send(state_event())
            elif data['action'] == 'diagnose':
                logging.info("got action diagnose message")
                if STATE['running'] and STATE['leads'] == 3: # end three lead ecg if running
                    DATA_GENERATOR_THREAD['thread'].do_run = False # stop thread
                    DATA_GENERATOR_THREAD['thread'].join()
                    DATA_GENERATOR_THREAD['thread'] = None
                    STATE['running'] = False
                    logging.debug("thread joined")
                if not STATE['running']: # start five lead ecg if not running
                    SIGNAL = [0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE
                    SIGNAL_TRANSFORM = [[0] * DIAGNOSE_DURATION * ADS1293.SAMPLE_RATE] * 4
                    DATA_GENERATOR_THREAD['thread'] = threading.Thread(target=generate_data, args=([5]))
                    DATA_GENERATOR_THREAD['thread'].start()
                    STATE['running'] = True
                    STATE['leads'] = 5
                    logging.debug("started thread")
                # start diagnostics
                SIGNALS[0] = []
                SIGNALS[1] = []
                SIGNALS[2] = []
                STATE['diagnosing'] = True
                STATE['diagnosestart'] = time.time()
                logging.debug("diagnosis started")
                await notify_state()
            else:
                logging.warning("unsupported event: {}".format(data))
    finally:
        await unregister(websocket)

def main():
    logging.basicConfig(level=logging.DEBUG)
    websocketslogger = logging.getLogger('websockets')
    websocketslogger.setLevel(logging.INFO)
    # get ads1293 data
    ads = ADS1293.ADS1293()
    ads.test()
    STATE['ina1max'] = ads.inax_max[0]
    STATE['ina1min'] = ads.inax_min[0]
    STATE['ina1zero'] = ads.inax_zero[0]
    STATE['ina2max'] = ads.inax_max[1]
    STATE['ina2min'] = ads.inax_min[1]
    STATE['ina2zero'] = ads.inax_zero[1]
    STATE['ina3max'] = ads.inax_max[2]
    STATE['ina3min'] = ads.inax_min[2]
    STATE['ina3zero'] = ads.inax_zero[2]
    ads.close()
    start_server = websockets.serve(listen, '', 9669)
    LOOP.run_until_complete(start_server)
    LOOP.run_forever()

if __name__ == "__main__":
    main()
