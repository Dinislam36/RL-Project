import pygame
import numpy as np


def rot_center(image, rect, angle):
    """rotate an image while keeping its center"""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect


# Process keys pressed
def process_keys(event, flag, move_up, move_down, move_left, move_right, shift):
    if event.key == pygame.K_LEFT:
        move_left = flag
    elif event.key == pygame.K_RIGHT:
        move_right = flag

    if event.key == pygame.K_UP:
        move_up = flag
    elif event.key == pygame.K_DOWN:
        move_down = flag

    if event.key == pygame.K_LSHIFT:
        shift = flag

    return move_up, move_down, move_left, move_right, shift