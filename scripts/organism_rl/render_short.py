import argparse
import random
import shutil
import subprocess
from pathlib import Path

import pygame

from agent import Agent
from food import Food
from organism import Organism


FPS = 30
CANVAS_SIZE = (720, 1280)
WORLD_SIZE = (800, 600)
WORLD_RECT = pygame.Rect(60, 380, 600, 450)
BACKGROUND = (13, 18, 32)
PANEL = (26, 35, 58)
TEXT = (240, 244, 255)
MUTED = (164, 180, 211)
GREEN = (65, 225, 142)
ORANGE = (255, 180, 82)
BLUE = (78, 151, 255)
CHECKPOINTS = (
    ('death_000500.pt', '500 ölüm', '%14 yem bulma', 5),
    ('death_002500.pt', '2.500 ölüm', '%75 yem bulma', 5),
    ('death_005000.pt', '5.000 ölüm', '%100 yem bulma', 5),
)


def draw_text(surface, font, text, color, center_x, top):
    rendered = font.render(text, True, color)
    surface.blit(rendered, (center_x - rendered.get_width() // 2, top))


def make_food(screen, organism, index):
    food = Food(screen, *WORLD_SIZE)
    offsets = ((110, 0), (0, 110), (-110, 0), (0, -110), (80, 80))
    offset_x, offset_y = offsets[index % len(offsets)]
    food.x = max(food.radius, min(WORLD_SIZE[0] - food.radius, organism.x + offset_x))
    food.y = max(food.radius, min(WORLD_SIZE[1] - food.radius, organism.y + offset_y))
    return food


def render_intro(surface, fonts):
    surface.fill(BACKGROUND)
    draw_text(surface, fonts['title'], 'Bir organizma', TEXT, 360, 350)
    draw_text(surface, fonts['title'], 'yem bulmayı öğreniyor', TEXT, 360, 418)
    draw_text(surface, fonts['subtitle'], 'Aynı görev • farklı eğitim aşamaları', MUTED, 360, 510)
    pygame.draw.circle(surface, (255, 77, 77), (300, 680), 42)
    pygame.draw.circle(surface, GREEN, (500, 680), 14)
    draw_text(surface, fonts['small'], 'Kırmızı: organizma     Yeşil: yem', MUTED, 360, 770)


def render_outro(surface, fonts):
    surface.fill(BACKGROUND)
    draw_text(surface, fonts['title'], 'Nasıl öğrendi?', TEXT, 360, 270)
    lines = (
        'Actor–Critic reinforcement learning',
        'Girdi: yem yönü, enerji, duvar uzaklıkları',
        'Ödül: yem +10 • yaklaşma + • ölüm -10',
        '100.000 ölüm • düzenli checkpoint testi',
    )
    for index, line in enumerate(lines):
        draw_text(surface, fonts['body'], line, GREEN if index == 0 else TEXT, 360, 410 + index * 88)
    draw_text(surface, fonts['small'], 'evolving-simple-organisms', MUTED, 360, 920)


def render_segment(surface, fonts, checkpoint, frame_count, frames_directory):
    filename, title, result, seconds = checkpoint
    agent = Agent(input_dim=7, output_dim=4, log_dir=None)
    agent.load_policy(Path(__file__).with_name('checkpoints') / filename)
    world = pygame.Surface(WORLD_SIZE)
    organism = Organism(world, *WORLD_SIZE, agent=agent)
    organism.x, organism.y, organism.energy, organism.score = 120, 300, 100, 0
    food_index = 0
    food = make_food(world, organism, food_index)
    action = 3
    action_steps = 0
    for frame in range(seconds * FPS):
        if action_steps == 0:
            action, _, _, _ = agent.select_action(organism.sense_food(food), deterministic=True)
            action_steps = 5
        organism.execute_action(action)
        action_steps -= 1
        if organism.food_distance(food) <= organism.radius + food.radius:
            food_index += 1
            food = make_food(world, organism, food_index)
            organism.energy += 10
            organism.score += 10
            action_steps = 0
        if organism.energy <= 0:
            organism.x, organism.y, organism.energy, organism.score = 120, 300, 100, 0
            food = make_food(world, organism, food_index)
            action_steps = 0

        surface.fill(BACKGROUND)
        draw_text(surface, fonts['title'], title, TEXT, 360, 86)
        draw_text(surface, fonts['subtitle'], result, GREEN, 360, 166)
        pygame.draw.rect(surface, PANEL, WORLD_RECT, border_radius=18)
        world.fill((245, 248, 255))
        organism.draw()
        food.draw()
        scaled_world = pygame.transform.smoothscale(world, WORLD_RECT.size)
        surface.blit(scaled_world, WORLD_RECT.topleft)
        progress = (frame + 1) / (seconds * FPS)
        pygame.draw.rect(surface, (65, 78, 112), (90, 910, 540, 10), border_radius=5)
        pygame.draw.rect(surface, GREEN, (90, 910, int(540 * progress), 10), border_radius=5)
        draw_text(surface, fonts['body'], f'Puan: {organism.score:.0f}', TEXT, 360, 990)
        draw_text(surface, fonts['small'], 'Aynı başlangıç koşulu • deterministik izleme', MUTED, 360, 1060)
        pygame.image.save(surface, frames_directory / f'frame_{frame_count + frame:05d}.png')
    return frame_count + seconds * FPS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Render a vertical learning-progress video.')
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).with_name('runs') / 'shorts' / 'one_food_learning_short.mp4',
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    random.seed(7)
    pygame.init()
    surface = pygame.Surface(CANVAS_SIZE)
    fonts = {
        'title': pygame.font.SysFont('DejaVu Sans', 44, bold=True),
        'subtitle': pygame.font.SysFont('DejaVu Sans', 28, bold=True),
        'body': pygame.font.SysFont('DejaVu Sans', 25),
        'small': pygame.font.SysFont('DejaVu Sans', 20),
    }
    frames_directory = arguments.output.with_suffix('')
    frames_directory = frames_directory.parent / f'{frames_directory.name}_frames'
    if frames_directory.exists():
        shutil.rmtree(frames_directory)
    frames_directory.mkdir(parents=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    for _ in range(2 * FPS):
        render_intro(surface, fonts)
        pygame.image.save(surface, frames_directory / f'frame_{frame_count:05d}.png')
        frame_count += 1
    for checkpoint in CHECKPOINTS:
        frame_count = render_segment(surface, fonts, checkpoint, frame_count, frames_directory)
    for _ in range(4 * FPS):
        render_outro(surface, fonts)
        pygame.image.save(surface, frames_directory / f'frame_{frame_count:05d}.png')
        frame_count += 1

    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(FPS), '-i', str(frames_directory / 'frame_%05d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(arguments.output),
    ], check=True)
    shutil.rmtree(frames_directory)
    pygame.quit()
    print(arguments.output)


if __name__ == '__main__':
    main()
