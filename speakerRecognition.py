import encoder.inference as encoder
import encoder.audio as audio
import numpy as np
import os
from pathlib import Path
import pyaudio
import threading
from typing import Callable, Optional
from queue import Queue
import time


class ByteCircularBuffer:
	"""
	A circular buffer for bytes with a fixed length.
	
	Attributes:
		_dataLen (int): The length of the buffer.
		_data (bytes): The buffer storing the byte data.
	"""

	def __init__(self, length: int, initialByte:bytes=None):
		self._dataLen = length
		if not initialByte:
			initialByte = b'\x00'
		self._data = initialByte * self._dataLen

	def put(self, newData: bytes) -> None:
		_l = len(newData)
		if _l <= self._dataLen:
			self._data = self._data[_l:] + newData


	def get_buffer(self) -> bytes:
		return self._data

	def get_buffer_len(self) -> int:
		return self._dataLen



class Acquisition:
	"""
	Acquisition class to handle audio streaming and processing.
	"""

	def __init__(self, fs: int, buffer_size: Optional[int] = 1024) -> None:
		self._buffer_size = buffer_size if buffer_size else 1024
		self._samplerate = fs
		self._acquisition_active = False
		self._queue = None

	def _acquisition(self) -> None:
		self._acquisition_active = True
		pya = pyaudio.PyAudio()

		stream = pya.open(
			format=pyaudio.paFloat32,
			channels=1,
			rate=self._samplerate,
			input=True,
			output=False,
			frames_per_buffer=self._buffer_size,
		)

		try:
			while self._acquisition_active:
				data = stream.read(self._buffer_size)
				if self._queue:
					self._queue.put(data)


		finally:
			stream.stop_stream()
			stream.close()
			pya.terminate()
			self._acquisition_active = False

	def start(self, queue_out:Queue) -> None:
		if not self._acquisition_active:
			self._acquisition_active = True
			self._queue = queue_out

			threading.Thread(target=self._acquisition, daemon=True).start()
		else:
			raise RuntimeError("Stream is already active.")

	def stop(self) -> None:
		if self._acquisition_active:
			self._acquisition_active = False
		else:
			raise RuntimeError("Stream is not active.")
	
	def is_stream(self) -> bool:
		return self._acquisition_active

	def get_type(self) -> type:
		return np.float32


class Process:
    """
    Process class to handle audio data processing using a pretrained model.
    """

    def __init__(self, buffer_size: int) -> None:
        print("Load GE2E pretrained model.")
        pretrained_model_path = Path(os.path.join(os.path.dirname(__file__), 'pretrained.pt'))
        encoder.load_model(pretrained_model_path)
        self._process_active = False
        self._buffer = ByteCircularBuffer(buffer_size)
        self._queue_in: Optional[Queue] = None
        self._queue_out: Optional[Queue] = None

    def _process_data(self, data: bytes) -> np.ndarray:
        self._buffer.put(data)
        npy_data = np.frombuffer(self._buffer.get_buffer(), dtype=np.float32)
        embedding = encoder.embed_utterance(npy_data, False, False)
        return embedding

    def _process_loop(self) -> None:
        if not encoder.is_loaded():
            raise RuntimeError("GE2E model is not loaded.")

        while self._process_active:
            if not self._queue_in.empty():
                data = self._queue_in.get()
                result = self._process_data(data)
                self._queue_out.put(result)

    def start(self, queue_in: Queue, queue_out: Queue) -> None:
        if not self._process_active:
            self._process_active = True
            self._queue_in = queue_in
            self._queue_out = queue_out
            threading.Thread(target=self._process_loop, daemon=True).start()
        else:
            raise RuntimeError("Process is already active.")

    def stop(self) -> None:
        if self._process_active:
            self._process_active = False
        else:
            raise RuntimeError("Process is not active.")

    def is_process(self) -> bool:
        return self._process_active


class SpeakerRecognition:
	def __init__(self, fs, buffer_size=None) -> None:
		self._acQueue = Queue(maxsize=5)
		self._acquisition = Acquisition(fs, buffer_size)
		self._processor = Process(int(fs*0.5))
		self._is_running = False

	def start(self, queue:Queue):
		self._is_running = True
		self._acquisition.start(self._acQueue)
		self._processor.start(self._acQueue, queue)

	def stop(self):
		self._is_running = False
		self._acquisition.stop()
		self._processor.stop()

	def is_running(self):
		return self._is_running


if __name__ == "__main__":
	guiQueue = Queue(maxsize=5)
	srCore = SpeakerRecognition(16000)
	srCore.start(guiQueue)
	time.sleep(1)

	srCore.stop()


