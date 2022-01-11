import pyaudio
import numpy as np


class Record:

    def __init__(self, fs, chunk_time):
        self.FORMAT = pyaudio.paInt16
        self.MAX_INT = 2 ** (16 - 1) - 1
        self.CHANNELS = 1
        self.FS = fs
        self.CHUNK_TIME = chunk_time
        self.CHUNK = round(self.FS * self.CHUNK_TIME)
        self.stream = None
        self.p = pyaudio.PyAudio()      

    def setup(self, device_id = 0):
        
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.FS,
                                  input=True,
                                  frames_per_buffer=self.CHUNK
                                  )

    def read(self):
        signal_byte = self.stream.read(self.CHUNK, exception_on_overflow = False)
        signal = np.frombuffer(signal_byte, dtype=np.int16) / self.MAX_INT
        return signal

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def get_id_device(self):
        # wypisanie informacji o urzadzeniach audio
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        id_device = 0
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                devicename = self.p.get_device_info_by_host_api_device_index(0, i).get('name')

                if devicename.startswith('seeed-8mic-voicecard'):
                    id_device = i
                print("Input Device id ", i, " - ", devicename.startswith('seeed-8mic-voicecard'))
        return id_device


if __name__ == '__main__':
    pass