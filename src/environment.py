import pygame
import numpy as np
import gym
from gym.spaces import Discrete, Box, MultiBinary
from src.pygame_utils import Hero, Hitbox, Bullet, play, create_n_bullets, check_collision, collide_hitbox_with_bullets


class BOWAPEnv(gym.Env):
    def __init__(self):

        # Preload img for faster bullet spawning
        pygame.init()
        self.sc = pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.img = pygame.image.load("../assets/bullet.png")
        # Action space
        self.action_space = Discrete(17)

        # Action dict. Map action to pressed keys:
        # int -> Up, Down, Left, Right, Shift
        self.action_dict = {
            0: (False, False, False, False, False),
            1: (False, False, True, False, False),
            2: (False, False, False, True, False),
            3: (True, False, False, False, False),
            4: (False, True, False, False, False),
            5: (False, False, True, False, True),
            6: (False, False, False, True, True),
            7: (True, False, False, False, True),
            8: (False, True, False, False, True),
            9: (True, False, True, False, False),
            10: (False, True, True, False, False),
            11: (True, False, False, True, False),
            12: (False, True, False, True, False),
            13: (True, False, True, False, True),
            14: (False, True, True, False, True),
            15: (True, False, False, True, True),
            16: (False, True, False, True, True)
        }

        # Playfield size
        self.max_x = 576
        self.max_y = 672
        self.max_dist = np.sqrt(self.max_x ** 2 + self.max_y ** 2)

        # How many bullets observation will take into account
        self.n_nearest_bullets = 512
        # Array for box high
        # high = [self.max_x if i % 2 == 0 else self.max_y for i in range(2 + self.n_nearest_bullets * 2)]
        # Define observation space
        self.observation_space = self.observation_space = Box(
            low=0, high=255, shape=(self.max_y, self.max_x, 3), dtype=np.uint8)

        # Create bullets group
        self.bullets = pygame.sprite.Group()

        # Hitbox info
        self.init_hitbox_pos_x = self.max_x // 2
        self.init_hitbox_pos_y = self.max_y - 24 * 4
        self.hitbox_size = 4

        # Hero images to render
        self.hero_surface = pygame.Surface((72, 48))
        self.reimu = Hero((24, 36))
        self.hitbox = Hitbox((24, 42))
        self.hero_surface.blit(self.reimu.image, self.reimu.rect)
        self.hero_surface.blit(self.hitbox.image, self.hitbox.rect)

        # Deaths
        self.deaths = 0

        # Bullet info
        self.bullet_spawn_pos_x = self.max_x // 2
        self.bullet_spawn_pos_y = self.max_y // 4
        self.bullet_size = 4

        # Bullet info
        self.bullet_offset = -4.5
        self.bullet_offset_speed = 0.0
        self.bullet_offset_acceleration = 0.225

        # Hero speed
        self.hero_speed_unfocused = 5 * 24 // 16
        self.hero_speed_focused = 2 * 24 // 16

        # Timer
        self.max_frame = 60 * 45 - 1
        self.frame = 0

        # Template for state
        self.state_template = np.array([self.bullet_spawn_pos_x if i % 2 == 0 else self.bullet_spawn_pos_y for i in
                                        range(2 + self.n_nearest_bullets * 2)], dtype=np.int32)

        # Logging movements
        self.log = []

        # Current state
        self.state = self.state_template.copy()
        self.state[0] = self.init_hitbox_pos_x
        self.state[1] = self.init_hitbox_pos_y

        # Reward increases as player lives
        self.frames_from_last_death = 0

    def step(self, action):
        # Get pressed keys from action
        up, down, left, right, shift = self.action_dict[action]

        # If both up and down are pressed, move up
        if up and down:
            down = False

        # If both left and right are pressed, move left
        if left and right:
            right = False

        # Append pressed keys to log
        self.log.append((up, down, left, right, shift))

        # How many frames ago was the last death
        self.frames_from_last_death += 1

        # Create bullets every 2 frames
        if self.frame % 2 == 0:
            self.__create_bullets_step()

        # Move bullets
        self.bullets.update()

        # Move hero
        self.__move_hero(up, down, left, right, shift)

        # Get array of moved bullets
        updated_bullets = np.array(self.__bullets_to_array(), dtype=np.int32)

        # Update state
        self.state[2:] = self.state_template[2:]
        self.state[2:2 + len(updated_bullets)] = updated_bullets

        # Get closest dist (was used previously as reward) and collision flag
        closest_dist, collide_flag = self.__collide_hitbox_with_bullets()
        # reward = closest_dist / self.max_dist

        # Reward increases as player survives
        reward = 0.1

        if action != 0:
            reward /= 2
        # reward = 1

        # If collision happened
        if collide_flag:
            # Clear bullets
            self.bullets.empty()
            # Increase death count
            self.deaths += 1
            # Set increasing reward to 0
            self.frames_from_last_death = 0
            # Death reward -100
            reward = -100

        # Done if timer gone
        done = True if self.frame >= self.max_frame else False

        info = {}

        self.frame += 1
        return self.__render_state_to_img(self.state[:2], self.bullets), reward, done, info

    def __render_state_to_img(self, hitbox_coords, bullet_group):
        playfield = pygame.Surface((self.max_y, self.max_x))
        playfield.fill((0, 0, 0, 0))
        hero_rect = self.hero_surface.get_rect(center=(hitbox_coords[1], hitbox_coords[0]))
        # Paste images on playfield
        playfield.blit(self.hero_surface, hero_rect)
        bullet_group.draw(playfield)
        return pygame.surfarray.array3d(playfield)

    def render(self):
        pass

    # Play logged replay
    def play_replay(self):
        play(self.log)

    # Scale state to 0,1 interval for neural networks
    def __scale_state(self, state):
        state = state.astype(np.float32)
        for i, val in enumerate(state):
            state[i] = val / self.max_x if i % 2 == 0 else val / self.max_y
        return state

    # Reset env
    def reset(self):
        # Set initial state
        self.state = self.state_template.copy()
        self.state[0] = self.init_hitbox_pos_x
        self.state[1] = self.init_hitbox_pos_y

        # Clear bullets
        self.bullets.empty()
        # Clear deaths
        self.deaths = 0
        # Set timer to 0
        self.frame = 0
        # Clear log
        self.log = []

        # Clear bullet spawn
        self.bullet_offset = -4.5
        self.bullet_offset_speed = 0.0
        self.bullet_offset_acceleration = 0.225

        return self.__render_state_to_img(self.state[:2], self.bullets)

    def __move_hero(self, move_up, move_down, move_left, move_right, shift):
        """Move hero according to action"""
        # Hero movement
        if self.state[1] - self.hitbox_size < 0:
            pass
        elif move_up and shift:
            self.state[1] -= self.hero_speed_focused
        elif move_up:
            self.state[1] -= self.hero_speed_unfocused

        if self.state[1] + self.hitbox_size > self.max_y:
            pass
        elif move_down and shift:
            self.state[1] += self.hero_speed_focused
        elif move_down:
            self.state[1] += self.hero_speed_unfocused

        if self.state[0] + self.hitbox_size > self.max_x:
            pass
        elif move_right and shift:
            self.state[0] += self.hero_speed_focused
        elif move_right:
            self.state[0] += self.hero_speed_unfocused

        if self.state[0] - self.hitbox_size < 0:
            pass
        elif move_left and shift:
            self.state[0] -= self.hero_speed_focused
        elif move_left:
            self.state[0] -= self.hero_speed_unfocused

    def __create_bullets_step(self):
        """Create n bullets from bullet spawn position"""
        self.bullet_offset += self.bullet_offset_speed
        self.bullet_offset %= 360
        self.bullet_offset_speed += self.bullet_offset_acceleration
        create_n_bullets((self.bullet_spawn_pos_y, self.bullet_spawn_pos_x),
                         6,
                         self.bullet_offset,
                         5,
                         self.bullets,
                         self.max_x,
                         self.max_y,
                         render=False,
                         loaded_img=self.img)

    def __check_collision(self, bullet_center, bullet_radius, hitbox_center, hitbox_radius):
        """Check collision of hitbox and single bullet"""
        b_c_x, b_c_y = bullet_center
        h_c_x, h_c_y = hitbox_center
        dist = np.sqrt(np.square(b_c_x - h_c_x) + np.square(b_c_y - h_c_y))
        if dist < bullet_radius + hitbox_radius:
            return dist, True
        return dist, False

    def __collide_hitbox_with_bullets(self):
        """Check collision of hitbox and bullets"""
        hitbox_pos_y = self.state[0]
        hitbox_pos_x = self.state[1]
        min_d = 10000
        for bullet in self.bullets:
            d, flag = self.__check_collision(bullet.collision_circle_center,
                                             bullet.collision_circle_radius,
                                             (hitbox_pos_x, hitbox_pos_y),
                                             self.hitbox_size)
            if d < min_d:
                min_d = d
            if flag:
                return d, True
        return min_d, False

    def __bullets_to_array(self):
        """Convert bullets sprite group to list of x and y positions"""
        out_arr = []
        for bullet in self.bullets:
            out_arr.append(bullet.rect.centerx)
            out_arr.append(bullet.rect.centery)
        return out_arr