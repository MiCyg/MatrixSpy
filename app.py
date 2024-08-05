import os
import wave
import time
import threading
import tkinter as tk
import numpy as np
from queue import Queue


import numpy as np
from speakerRecognition import SpeakerRecognition

class App:
	def __init__(self):
		self.root = tk.Tk()
		self.root.resizable(False, False)

		self.frame = tk.Label(text="Speaker Recognition")
		self.frame.pack()

		self.recordButton = tk.Button(text="▶️", font=("Arial", 60), command=self.click_handler)
		self.recordButton.pack()

		self.guiQueue = Queue(maxsize=5)
		self.srCore = SpeakerRecognition(16000, 1024*4)

	def click_handler(self):
		if self.srCore.is_running():
			self.recordButton.config(fg="Red")
			self.srCore.stop()
		else:
			self.recordButton.config(fg="Black")
			self.srCore.start(self.guiQueue)


	def run(self):
		self.root.mainloop()


