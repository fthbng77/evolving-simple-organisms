import argparse
import random
import shutil
import subprocess
from pathlib import Path

import pygame
import torch

from agent import Agent
from food import Food
from multi_main import ORGANISM_COLORS, choose_target, create_foods, recolor_foods
from organism import Organism
from scenario import load_scenario


FPS = 30
CANVAS_SIZE = (720, 1280)
WORLD_SIZE = (800, 600)
WORLD_RECT = pygame.Rect(60, 360, 600, 450)
BACKGROUND = (13, 18, 32)
PANEL = (26, 35, 58)
TEXT = (240, 244, 255)
MUTED = (164, 180, 211)
GREEN = (65, 225, 142)
CHECKPOINTS = (
    ('death_000500.pt', '500 ölüm', 'Başta ortak politikaları\nkararsız.', 6),
    ('death_002500.pt', '2.500 ölüm', 'Farklı konumlar, aynı beyin.\nHedefe yönelim artıyor.', 6),
    ('death_005000.pt', '5.000 ölüm', 'Yem toplama belirgin biçimde\nhızlanıyor.', 6),
    ('death_100000.pt', '100.000 ölüm', 'Her organizma düzenli olarak\nyem bulabiliyor.', 6),
)


def draw_center(surface, font, text, color, center_x, top):
    rendered = font.render(text, True, color)
    surface.blit(rendered, (center_x - rendered.get_width() // 2, top))


def draw_subtitle(surface, fonts, lines):
    pygame.draw.rect(surface, (5, 8, 16), (45, 1040, 630, 125), border_radius=18)
    for index, line in enumerate(lines.split('\n')):
        draw_center(surface, fonts['body'], line, TEXT, 360, 1062 + index * 38)


def draw_brain(surface, center):
    x, y = center
    nodes = ((x - 20, y - 16), (x - 20, y + 16), (x + 18, y), (x + 48, y))
    for start, end in zip(nodes, nodes[1:]):
        pygame.draw.line(surface, GREEN, start, end, 3)
    for node in nodes:
        pygame.draw.circle(surface, GREEN, node, 7)


def draw_intro(surface, fonts):
    surface.fill(BACKGROUND)
    draw_center(surface, fonts['title'], '3 ORGANİZMA', TEXT, 360, 240)
    draw_center(surface, fonts['title'], '1 ORTAK BEYİN', GREEN, 360, 310)
    draw_brain(surface, (305, 470))
    pygame.draw.circle(surface, ORGANISM_COLORS[0], (200, 630), 28)
    pygame.draw.circle(surface, ORGANISM_COLORS[1], (360, 630), 28)
    pygame.draw.circle(surface, ORGANISM_COLORS[2], (520, 630), 28)
    draw_center(surface, fonts['small'], 'Aynı politika • farklı konumlar', MUTED, 360, 740)
    draw_subtitle(surface, fonts, 'Üç organizma aynı ortamda\nyem bulmayı öğreniyor.')


def draw_outro(surface, fonts):
    surface.fill(BACKGROUND)
    draw_center(surface, fonts['title'], 'NASIL ÖĞRENDİ?', TEXT, 360, 260)
    lines = (
        'Paylaşımlı Actor–Critic',
        'Girdi: yem, enerji, duvarlar, diğer canlı',
        'Ödül: yem +10 • yaklaşma + • ölüm -10',
    )
    for index, line in enumerate(lines):
        draw_center(surface, fonts['body'], line, GREEN if index == 0 else TEXT, 360, 410 + index * 82)
    draw_center(surface, fonts['small'], 'Her canlı farklı hareket eder; beyin aynıdır.', MUTED, 360, 760)
    draw_subtitle(surface, fonts, 'Yöntem: paylaşımlı Actor–Critic\npekiştirmeli öğrenme.')


def write_frame(surface, frames_directory, frame_number):
    pygame.image.save(surface, frames_directory / f'frame_{frame_number:05d}.png')


def render_segment(surface, fonts, checkpoint, frame_number, frames_directory, run_directory, scenario):
    filename, title, subtitle, seconds = checkpoint
    agent = Agent(input_dim=10, output_dim=4, device=torch.device('cpu'), log_dir=None)
    agent.load_policy(run_directory / 'checkpoints' / filename)
    world = pygame.Surface(WORLD_SIZE)
    organisms = [
        Organism(world, *WORLD_SIZE, agent=agent, color=ORGANISM_COLORS[index % len(ORGANISM_COLORS)])
        for index in range(scenario.organism_count)
    ]
    starts = ((130, 160), (390, 300), (650, 460))
    for organism, (x, y) in zip(organisms, starts):
        organism.x, organism.y, organism.energy, organism.score = x, y, 100, 0
    foods = create_foods(world, organisms, scenario)
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    actions = [0] * scenario.organism_count
    action_steps = [0] * scenario.organism_count
    trails = [[(organism.x, organism.y)] for organism in organisms]
    floating_scores = []
    deaths = [0] * scenario.organism_count

    for frame in range(seconds * FPS):
        for index, organism in enumerate(organisms):
            if action_steps[index] == 0:
                state = organism.sense_environment(targets[organism], organisms)
                actions[index], _, _, _ = agent.select_action(state, deterministic=True)
                action_steps[index] = scenario.action_repeat
            organism.execute_action(actions[index])
            action_steps[index] -= 1
            trails[index].append((organism.x, organism.y))
            trails[index] = trails[index][-32:]
            eaten_food = next(
                (food for food in foods if organism.food_distance(food) <= organism.radius + food.radius),
                None,
            )
            if eaten_food:
                floating_scores.append([eaten_food.x, eaten_food.y, 24, organism.color])
                foods.remove(eaten_food)
                foods.append(Food.near_organism(
                    world, *WORLD_SIZE, organism,
                    scenario.minimum_food_distance, scenario.maximum_food_distance,
                ))
                for other in organisms:
                    if targets[other] is eaten_food:
                        targets[other] = choose_target(other, foods)
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                organism.energy += 10
                organism.score += 10
                action_steps[index] = 0
            if organism.energy <= 0:
                organism.x, organism.y, organism.energy, organism.score = *starts[index], 100, 0
                targets[organism] = choose_target(organism, foods)
                action_steps[index] = 0
                deaths[index] += 1

        surface.fill(BACKGROUND)
        draw_center(surface, fonts['title'], title, TEXT, 360, 72)
        draw_center(surface, fonts['subtitle'], 'Aynı ortam • aynı başlangıç • farklı eğitim aşaması', MUTED, 360, 140)
        pygame.draw.rect(surface, PANEL, WORLD_RECT, border_radius=18)
        world.fill((245, 248, 255))
        for path, color in zip(trails, ORGANISM_COLORS):
            if len(path) > 1:
                pygame.draw.lines(world, color, False, path, 3)
        for food in foods:
            food.draw()
        for organism in organisms:
            target = targets[organism]
            pygame.draw.circle(world, organism.color, (int(target.x), int(target.y)), target.radius + 6, 2)
        for organism in organisms:
            organism.draw()
        for score in floating_scores[:]:
            x, y, ttl, color = score
            rendered = fonts['world'].render('+10', True, color)
            world.blit(rendered, (x - 16, y - 12 - (24 - ttl) * 2))
            score[2] -= 1
            if score[2] <= 0:
                floating_scores.remove(score)
        surface.blit(pygame.transform.smoothscale(world, WORLD_RECT.size), WORLD_RECT.topleft)
        progress = (frame + 1) / (seconds * FPS)
        pygame.draw.rect(surface, (65, 78, 112), (90, 880, 540, 10), border_radius=5)
        pygame.draw.rect(surface, GREEN, (90, 880, int(540 * progress), 10), border_radius=5)
        scores = '  '.join(f'{index + 1}: {organism.score:.0f}' for index, organism in enumerate(organisms))
        draw_center(surface, fonts['body'], f'Puanlar  {scores}', TEXT, 360, 940)
        draw_center(surface, fonts['small'], 'Hedef halkası = o organizmanın aktif hedefi', MUTED, 360, 990)
        draw_subtitle(surface, fonts, subtitle)
        write_frame(surface, frames_directory, frame_number + frame)
    return frame_number + seconds * FPS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Render a multi-organism vertical learning video.')
    parser.add_argument(
        '--run', type=Path,
        default=Path(__file__).with_name('runs') / 'multi_organism' / '20261015-143417',
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).with_name('runs') / 'shorts' / 'multi_organism_learning_short.mp4',
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    scenario = load_scenario(arguments.run / 'scenario.json')
    random.seed(19)
    pygame.init()
    surface = pygame.Surface(CANVAS_SIZE)
    fonts = {
        'title': pygame.font.SysFont('DejaVu Sans', 42, bold=True),
        'subtitle': pygame.font.SysFont('DejaVu Sans', 20),
        'body': pygame.font.SysFont('DejaVu Sans', 24),
        'small': pygame.font.SysFont('DejaVu Sans', 18),
        'world': pygame.font.SysFont('DejaVu Sans', 24, bold=True),
    }
    frames_directory = arguments.output.parent / f'{arguments.output.stem}_frames'
    if frames_directory.exists():
        shutil.rmtree(frames_directory)
    frames_directory.mkdir(parents=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    for _ in range(3 * FPS):
        draw_intro(surface, fonts)
        write_frame(surface, frames_directory, frame_number)
        frame_number += 1
    for checkpoint in CHECKPOINTS:
        frame_number = render_segment(
            surface, fonts, checkpoint, frame_number, frames_directory, arguments.run, scenario,
        )
    for _ in range(4 * FPS):
        draw_outro(surface, fonts)
        write_frame(surface, frames_directory, frame_number)
        frame_number += 1

    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(FPS), '-i', str(frames_directory / 'frame_%05d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(arguments.output),
    ], check=True)
    shutil.rmtree(frames_directory)
    pygame.quit()
    print(arguments.output)


if __name__ == '__main__':
    main()
