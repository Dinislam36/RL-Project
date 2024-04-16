import pygame
import numpy as np
from src.utils import rot_center, process_keys


# Create n bullets rotated equally on position
def create_n_bullets(pos, n, offset, speed, group, playfield_width, playfield_height, render=True, loaded_img=False):
    degree_between_bullets = 360 / n
    angles = [degree_between_bullets * i + offset for i in range(n)]

    bullets = []

    for angle in angles:
        Bullet(pos, angle, speed, group, playfield_width, playfield_height, render=render, loaded_img=loaded_img)


# Check cillision between circles
def check_collision(bullet_center, bullet_radius, hitbox_center, hitbox_radius):
    b_c_x, b_c_y = bullet_center
    h_c_x, h_c_y = hitbox_center
    dist = np.sqrt(np.square(b_c_x - h_c_x) + np.square(b_c_y - h_c_y))
    if dist < bullet_radius + hitbox_radius:
        return True
    return False


# Check collision between hitbox and all bullets
def collide_hitbox_with_bullets(hitbox, hero_rect, bullets):
    hitbox_pos_x = hero_rect.topleft[0] + hitbox.collision_circle_center[0]
    hitbox_pos_y = hero_rect.topleft[1] + hitbox.collision_circle_center[1]

    for bullet in bullets:
        if check_collision(bullet.collision_circle_center,
                           bullet.collision_circle_radius,
                           (hitbox_pos_x, hitbox_pos_y),
                           hitbox.collision_circle_radius):
            return True
    return False


class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos, angle, speed, group, playfield_width, playfield_height, collision_rect_width=6,
                 collision_rect_height=14, render=True, loaded_img=False):
        pygame.sprite.Sprite.__init__(self)

        # Load bullet image
        if loaded_img:
            image = loaded_img
        else:
            image = pygame.image.load("../assets/bullet.png")

        if render:
            image = image.convert_alpha()
        # Scale image
        image = pygame.transform.scale(image, (24, 24))
        # Get rectangle from image
        rect = image.get_rect(center=pos)
        # Rotate rectangle
        self.image, self.rect = rot_center(image, rect, angle)

        # Angle
        self.angle = angle

        # Collision circle center
        self.collision_circle_center = np.array(pos, dtype=np.float32)
        # Collision circle radius
        self.collision_circle_radius = 4

        # Playfield resolution
        self.playfield_width = playfield_width
        self.playfield_height = playfield_height

        # Speed of bullet
        self.speed_y = float(np.cos(np.deg2rad(angle))) * speed
        self.speed_x = float(np.sin(np.deg2rad(angle))) * speed
        # Position of the bullet (float)
        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)

        self.timer = 180  # How many frames bullet will alive
        self.add(group)

    def __update_collision_circle(self):
        # Move collision center
        self.collision_circle_center[0] += self.speed_x
        self.collision_circle_center[1] += self.speed_y

    def update(self):
        self.timer -= 1

        # Kill if out of the screen or time out
        if (self.rect.centerx <= 0 or
                self.rect.centery <= 0 or
                self.rect.centerx >= self.playfield_width or
                self.rect.centery >= self.playfield_height or
                self.timer == 0):
            self.kill()

        # Position of bullet (in float)
        self.pos_x += self.speed_x
        self.pos_y += self.speed_y

        # Assign rectangle position (int)
        self.rect.x = int(self.pos_x)
        self.rect.y = int(self.pos_y)

        self.__update_collision_circle()


class Hitbox(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)

        # Load hitbox image
        self.image = pygame.image.load("../assets/hitbox.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (15, 15))
        self.rect = pygame.Rect(32 - 7 + 6, 24 - 7, 15, 15)
        # Hitbox collision
        self.collision_circle_radius = 4
        self.collision_circle_center = (self.rect.centerx, self.rect.centery)


# Hero sprite
class Hero(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)

        image = pygame.image.load("../assets/reimu.png").convert_alpha()
        self.rect = pygame.Rect(0, 0, 72, 48)
        image = pygame.transform.rotate(image, 90)
        self.image = pygame.transform.scale(image, (72, 48))



def play(movement_logs=False):
    # Init game
    pygame.init()
    # pygame.time.set_timer(pygame.USEREVENT, 33)

    # Create font
    f = pygame.font.SysFont('arial', 30)

    # Playfield resolution in blocks
    th_playfield_resolution = (24, 28)
    # Scorefield width in blocks
    th_score_size = 14
    # Block size in pixels. Originally block size is 16
    block_size = 24
    # FPS
    FPS = 60

    # Colors
    LIGHT_BLUE = (135, 206, 235)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREED = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Shapes of fields
    W_playfield = th_playfield_resolution[0] * block_size
    W_scorefield = th_score_size * block_size
    H = th_playfield_resolution[1] * block_size

    # Hero size (original when block size = 16)
    hero_size = (2, 3)
    hero_resolution = (32, 48)
    hero_speed_unfocused = 5 * 24 // 16
    hero_speed_focused = 2 * 24 // 16

    # Hitbox size (original, when block size = 16)
    hitbox_size = (0.625, 0.625)
    hitbox_resolution = (10, 10)
    # Bullet size (original, when block size = 16)
    bullet_size = (1, 1)
    bullet_resolution = (16, 16)

    # Get main field
    sc = pygame.display.set_mode((W_playfield + W_scorefield, H))
    # Create playfield
    playfield = pygame.Surface((W_playfield, H))
    # Create scorefield
    scorefield = pygame.Surface((W_scorefield, H))
    # Draw scorefield to light blue
    pygame.draw.rect(scorefield, LIGHT_BLUE, (0, 0, W_scorefield, H))

    # Create hero and hitbox
    hero_surface = pygame.Surface((72, 48))
    reimu = Hero((24, 36))
    hitbox = Hitbox((24, 42))
    hitbox_center = (32, 50)

    # Paste images on hero surface
    hero_surface.blit(reimu.image, reimu.rect)
    hero_surface.blit(hitbox.image, hitbox.rect)

    hero_rect = hero_surface.get_rect(center=(W_playfield // 2, H - block_size * 4))
    # Paste images on playfield
    playfield.blit(hero_surface, hero_rect)

    # Create score text
    score = 0
    deaths = 0

    # Paste fields over main field
    sc.blit(playfield, (0, 0))
    sc.blit(scorefield, (W_playfield, 0))
    # Set name
    pygame.display.set_caption("BOWAP")
    # Update screen
    pygame.display.update()

    # Check if game s running
    running = True
    # Timer
    clock = pygame.time.Clock()
    # Bullet group
    bullets = pygame.sprite.Group()
    # Bullet spawn position
    bullet_spawn_pos = (th_playfield_resolution[0] // 2 * block_size, th_playfield_resolution[1] // 4 * block_size)
    # Initial bullet spawn angle
    offset = -4.5
    # Initial angle speed
    offset_speed = 0
    # Angle acceleration
    acceleration = 0.225

    # Hero movement
    hero_move_x = 0
    hero_move_y = 0

    # Hero movement flags
    move_up = False
    move_down = False
    move_left = False
    move_right = False
    shift = False

    # Game loop for 45 seconds
    for frame in range(FPS * 45):
        playfield.fill((0, 0, 0, 0))

        # Spawn bullets
        if frame % 2 == 0:
            offset += offset_speed
            offset %= 360
            offset_speed += acceleration
            create_n_bullets(bullet_spawn_pos, 6, offset, 5, bullets, W_playfield, H)
        # Render timer on top right corner
        if frame % 60 == 0:
            time_text = f.render(str((FPS * 45 - frame) // 60), True, WHITE)

        # Check for keys pressed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False


            elif event.type == pygame.KEYDOWN and not movement_logs:
                move_up, move_down, move_left, move_right, shift = process_keys(event,
                                                                                True,
                                                                                move_up,
                                                                                move_down,
                                                                                move_left,
                                                                                move_right,
                                                                                shift)

            elif event.type == pygame.KEYUP and not movement_logs:
                move_up, move_down, move_left, move_right, shift = process_keys(event,
                                                                                False,
                                                                                move_up,
                                                                                move_down,
                                                                                move_left,
                                                                                move_right,
                                                                                shift)

        # Hitbox features
        hitbox_radius = (hitbox.rect.width + 1) // 2
        hitbox_offset = 6

        # If there is logs
        if movement_logs:
            move_up, move_down, move_left, move_right, shift = movement_logs[frame]

        # Hero movement
        if hero_rect.centery - hitbox_radius + hitbox_offset < 0:
            pass
        elif move_up and shift:
            hero_rect.y -= hero_speed_focused
        elif move_up:
            hero_rect.y -= hero_speed_unfocused

        if hero_rect.centery + hitbox_radius > H:
            pass
        elif move_down and shift:
            hero_rect.y += hero_speed_focused
        elif move_down:
            hero_rect.y += hero_speed_unfocused

        if hero_rect.centerx + hitbox_radius > W_playfield:
            pass
        elif move_right and shift:
            hero_rect.x += hero_speed_focused
        elif move_right:
            hero_rect.x += hero_speed_unfocused

        if hero_rect.centerx - hitbox_radius < 0:
            pass
        elif move_left and shift:
            hero_rect.x -= hero_speed_focused
        elif move_left:
            hero_rect.x -= hero_speed_unfocused

        # Paste hero on playfield
        playfield.blit(hero_surface, hero_rect)
        # Draw bullets
        bullets.draw(playfield)
        # Paste timer
        playfield.blit(time_text, (W_playfield - 50, 10))

        # Check collision
        if collide_hitbox_with_bullets(hitbox, hero_rect, bullets):
            # Remove bullets on death
            bullets.empty()
            deaths += 1
            score = -100 * deaths

        # Score and death text
        score_text = f.render("Total score: " + str(score), True, WHITE)
        death_text = f.render("Deaths: " + str(deaths), True, WHITE)

        # Draw bg on scorefield
        pygame.draw.rect(scorefield, LIGHT_BLUE, (0, 0, W_scorefield, H))

        # Paste score and deaths on scorefield
        scorefield.blit(score_text, (10, H // 2 - 50))
        scorefield.blit(death_text, (10, H // 2))

        for bullet in bullets:
            pygame.draw.circle(playfield, LIGHT_BLUE, bullet.collision_circle_center, bullet.collision_circle_radius)
        # Paste playfield and scorefield on main screen
        sc.blit(playfield, (0, 0))
        sc.blit(scorefield, (W_playfield, 0))
        # Update main screen
        pygame.display.update()
        # Tick FPS
        clock.tick(FPS)
        # Update bullets
        bullets.update()

    # Quit on time out
    if running:
        pygame.quit()