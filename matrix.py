
# set up matrix
from luma.core.render import canvas
from luma.led_matrix.device import ws2812
from luma.core.legacy import text
from luma.core.legacy.font import proportional, CP437_FONT, LCD_FONT, ATARI_FONT, SEG7_FONT, SINCLAIR_FONT
from rpi_ws281x import *

class Matrix:

    def __init__(self):

        # LED strip configuration:
        LED_COUNT = 256      # Number of LED pixels.
        LED_PIN = 12      # GPIO pin connected to the pixels (must support PWM!).
        LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
        LED_DMA = 10       # DMA channel to use for generating signal (try 10)
        LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
        # True to invert the signal (when using NPN transistor level shift)
        LED_INVERT = False
        WIDTH = 32
        HEIGHT = 8


        #strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)

        MY_MAPPING = [0, 15, 16, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 111, 112, 127, 128, 143, 144, 159, 160, 175, 176, 191, 192, 207, 208, 223, 224, 239,240, 255, 1, 14, 17, 30, 33, 46, 49, 62, 65, 78, 81, 94, 97, 110, 113, 126, 129, 142, 145, 158, 161, 174, 177, 190, 193, 206, 209, 222,225, 238, 241, 254, 2, 13, 18, 29, 34, 45, 50, 61, 66, 77, 82, 93, 98, 109, 114, 125, 130, 141, 146, 157, 162, 173, 178, 189, 194, 205,210, 221, 226, 237, 242, 253, 3, 12, 19, 28, 35, 44, 51, 60, 67, 76, 83, 92, 99, 108, 115, 124, 131, 140, 147, 156, 163, 172, 179, 188,195, 204, 211, 220, 227, 236, 243, 252, 4, 11, 20, 27, 36, 43, 52, 59, 68, 75, 84, 91, 100, 107, 116, 123, 132, 139, 148, 155, 164, 171,180, 187, 196, 203, 212, 219, 228, 235, 244, 251, 5, 10, 21, 26, 37, 42, 53, 58, 69, 74, 85, 90, 101, 106, 117, 122, 133, 138, 149, 154,165, 170, 181, 186, 197, 202, 213, 218, 229, 234, 245, 250, 6, 9, 22, 25, 38, 41, 54, 57, 70, 73, 86, 89, 102, 105, 118, 121, 134, 137,150, 153, 166, 169, 182, 185, 198, 201, 214, 217, 230, 233, 246, 249, 7, 8, 23, 24, 39, 40, 55, 56, 71, 72, 87, 88, 103, 104, 119, 120,135, 136, 151, 152, 167, 168, 183, 184, 199, 200, 215, 216, 231, 232, 247, 248]

        self.device = ws2812(width=WIDTH, height=HEIGHT, mapping=MY_MAPPING)

    def text(self, msg, color='white', posx=0, posy=0):
        with canvas(self.device) as draw:
            text(draw, (posx, -posy), msg, fill=color)
        
        
        
if __name__ == '__main__':
    test_matrix = Matrix()
    test_matrix.text('hello!', color=(50,50,50))
