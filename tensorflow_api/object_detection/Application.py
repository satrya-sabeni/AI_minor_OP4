import pygame, sys
from pygame.locals import *

def main():
    pygame.init()

    DISPLAY=pygame.display.set_mode((800,600),0,32)

    WHITE=(255,255,255)
    BLUE=(0,0,255)



    while True:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()

            if event.type==QUIT or keys[pygame.K_q]:
                pygame.quit()
                sys.exit()
        mousepos = pygame.mouse.get_pos()
        movemouse(keys,mousepos)

        DISPLAY.fill(WHITE)

        pygame.display.update()

def movemouse(keys,mousepos):
    if(keys[pygame.K_LEFT]):
            pygame.mouse.set_pos(mousepos[0]-1,mousepos[1])

    if(keys[pygame.K_RIGHT]):
        pygame.mouse.set_pos(mousepos[0]+1,mousepos[1])

    if(keys[pygame.K_UP]):
        pygame.mouse.set_pos(mousepos[0],mousepos[1]-1)

    if(keys[pygame.K_DOWN]):
        pygame.mouse.set_pos(mousepos[0],mousepos[1]+1)

    if(keys[pygame.K_DOWN] and keys[pygame.K_LEFT]):
        pygame.mouse.set_pos(mousepos[0]-1,mousepos[1]+1)
    
    if(keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]):
        pygame.mouse.set_pos(mousepos[0]+1,mousepos[1]+1)

    if(keys[pygame.K_UP] and keys[pygame.K_RIGHT]):
        pygame.mouse.set_pos(mousepos[0]+1,mousepos[1]-1)

    if(keys[pygame.K_UP] and keys[pygame.K_LEFT]):
        pygame.mouse.set_pos(mousepos[0]-1,mousepos[1]-1)


main()