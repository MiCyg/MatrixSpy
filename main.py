

from speaker_verification import *
from record import Record
from matrix import Matrix
import time

from scipy.io.wavfile import write
from threading import Thread
import RPi.GPIO as GPIO



#================== AVERAGING NAMES ===================
def average_list(list_names, list_of_elements=None):
    # jeśli nie mamy żadnych danych odnośnie nazw, musimy sprawdzić jakie są
    if list_of_elements is None:
        list_of_elements = []
        for name in list_names:
            if name not in list_of_elements:
                list_of_elements.append(name)

    counts = {}
    for element in list_of_elements:
        counts[element] = list_names.count(element)

    out = max(counts, key=counts.get)
    return out


global show_name_global
show_name_global = 'Waiting...'
global sufix
sufix = '_train'

global RUN
RUN = True


def mainVR():

    
    number_of_average = 8
    moving_classifier = []
    for i in range(number_of_average):
        moving_classifier.append(None)
        
    # ====================== EMBEDDINGS =======================

    path_to_records = Path('rec')
    path_to_save_emb = Path('emb')
    embeddings = insert_speakers(path_to_records, path_to_save_emb, end_filename=sufix, verbose=1)


    speaker_names = list(embeddings.keys())
    speaker_names.append(None)

    # =================== LIVE RECORDING ====================
    fs = 16000
    SIGNAL_EMB_TIME = 1.6
    N_moving_signal = round(SIGNAL_EMB_TIME*fs)
    moving_signal = np.zeros(N_moving_signal)


    CHUNK_TIME = 0.4
    CHUNK = round(CHUNK_TIME*fs)
    rec = Record(fs=fs, chunk_time=CHUNK_TIME)
    id = rec.get_id_device()
    rec.setup(id)

    #================== MAIN LOOP ===================
    global show_name_global
    global RUN
    
    record_data = np.array([])
    i = 0
    print("* start embedding")
    while RUN:
        try:

            actual_signal = rec.read()
            record_data = np.append(record_data, actual_signal)

            # dodanie do bufora sygnału i wyciszenie krańców
            moving_signal = np.append(moving_signal[CHUNK:N_moving_signal], actual_signal)
            #moving_signal = fading(moving_signal, fs, timefold=0.05)

            # weryfikacja mówcy
            embedding = count_embedding(moving_signal, fs)
            speaker_filename = speaker_ver(embedding, embeddings, coeff=1)

            # uśrednienie wyników
            moving_classifier.pop(0)
            moving_classifier.append(speaker_filename)
            best_filename = average_list(moving_classifier)
            
            # wrzucenie nazwy do zmiennej globalnej
            #show_name_global = str(best_filename)
            show_name_global = str(speaker_filename)
            
            print(i, '->', speaker_filename)
            i += 1

        except ValueError:
            # errory występują w 0.45% przypadków (sprawdzane na godzinnym nagraniu)
            print('ERROR')
            #errors += 1



    print("* done embedding")

    rec.stop()

    convert_data = record_data*((2**16-1)-1)
    write("example.wav", fs, convert_data.astype(np.int16))


def mainM():
    print('start matrix thread')
    matrix = Matrix()
    WIDTH = 32
    HEIGTH = 8
    
    
    global show_name_global
    global sufix
    
    color = (50,50,50)
    delay = 0.1
    
    name = ''
    pos = WIDTH
    global RUN
    while RUN:
        
        if show_name_global == 'None':
            color = (50,0,0)
            name = show_name_global
        else:
            color = (50,0,50)
            name = show_name_global.replace(sufix + '.wav', '')
            
        matrix.text(name, color, posx=pos)
        pos -= 1
        if pos <= -(8*len(name)):
            pos = WIDTH
            
        time.sleep(delay)
        
        

if __name__ == '__main__':
    
    VoiceRecognitionThread = Thread(target=mainVR)
    MatrixThread = Thread(target=mainM)

    VoiceRecognitionThread.start()
    MatrixThread.start()
    
    # === MAIN PROGRAM ===
    BUTTON = 26
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON, GPIO.IN)
    
    state = False
    while True:
        
        state = GPIO.input(BUTTON)
        if not state:
            print('Interrupt application')
            RUN = False
            break
        
    VoiceRecognitionThread.terminate()
    MatrixThread.terminate()
    