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
		self.button = tk.Button(text="▶️", font=("Arial", 120), command=self.click_handler)
		self.button.pack()

		self.guiQueue = Queue(maxsize=5)
		self.srCore = SpeakerRecognition(16000, 1024*4)

	def click_handler(self):
		if self.srCore.is_running():
			self.button.config(fg="Red")
			self.srCore.stop()
		else:
			self.button.config(fg="Black")
			self.srCore.start(self.guiQueue)


	def run(self):
		self.root.mainloop()


