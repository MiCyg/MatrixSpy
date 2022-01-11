import sys
sys.path.append('../')

import numpy as np
from record import Record
from scipy.io.wavfile import write
import time

# =================== LIVE RECORDING ====================
fs = 16000
CHUNK_TIME = 0.4
CHUNK = round(CHUNK_TIME*fs)
rec = Record(fs=fs, chunk_time=CHUNK_TIME)
id = rec.get_id_device()
rec.setup(id)

#================== MAIN LOOP ===================
record_data = np.array([])
i = 0
print("* start embedding")
while True:
    try:

        actual_signal = rec.read()
        record_data = np.append(record_data, actual_signal)
        print(i*CHUNK_TIME)
        i+=1

    except ValueError:
        # errory występują w 0.45% przypadków (sprawdzane na godzinnym nagraniu)
        print('ERROR')
    except KeyboardInterrupt:
        break




convert_data = record_data*((2**16-1)-1)
write("output.wav", fs, convert_data.astype(np.int16))

print("* done embedding")

rec.stop()





