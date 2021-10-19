
import time
import sys
from gym_pacman.envs.pacmanenv import PacmanEnv
#from pacmanenv import PacmanEnv
import numpy as np
from mss import mss
import cv2
from numpy.core.fromnumeric import shape
import numpy as np
import pyautogui
from pygame.constants import HIDDEN
if __name__ == "__main__":
    env = PacmanEnv()
    i = 0
    bounding_box = {'top': 170 , 'left': 100, 'width': 448, 'height': 576-16*4}
    TOP = 170
    LEFT = 100
    WIDTH = 448
    HEIGHT = 576-16*4
    sct = mss()
    counter = 1 
    while 100:

        #img = np.array(sct.grab(bounding_box))
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        myScreenshot = pyautogui.screenshot(region=(TOP,LEFT,WIDTH,HEIGHT))
        open_cv_image = np.array(myScreenshot) 
        #myScreenshot.save(r'Path to save screenshot\file name.png')

        # cv2.imshow("test",open_cv_image)
        # cv2.waitKey(1)
        env.step(0)
        # Display the picture
        
        counter = counter + 1
        #print("score",game.score,"pacman pose" ,game.pacman.position,"Ghost", game.ghosts.ghosts[1].position,"ghost mode", game.ghosts.ghosts[1].mode.current, " AI event ",PACMAN_EVENTS_NAME[game.events_AI])
        #time.sleep(0.2)

        
                                                        


