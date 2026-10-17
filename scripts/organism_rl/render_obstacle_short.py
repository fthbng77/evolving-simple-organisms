import argparse
import random
import shutil
import subprocess
from pathlib import Path

import pygame
import torch

from agent import Agent
from multi_main import ORGANISM_COLORS, choose_target, recolor_foods
from obstacle_main import create_fields, create_foods, load_obstacles
from obstacles import move_with_obstacles, obstacle_distances, spawn_food
from organism import Organism
from scenario import load_scenario


FPS = 30
CANVAS = (720, 1280)
WORLD = (800, 600)
WORLD_RECT = pygame.Rect(60, 350, 600, 450)
BACKGROUND, PANEL, TEXT, MUTED = (13, 18, 32), (26, 35, 58), (240, 244, 255), (164, 180, 211)
GREEN, OBSTACLE = (65, 225, 142), (75, 84, 105)
CHECKPOINTS = (
    ('death_000500.pt', '500 ÖLÜM', 'Başta doğrudan hedefe gidiyor;\nengel çarpışmaları sık.', 6),
    ('death_002500.pt', '2.500 ÖLÜM', 'Sekiz yönlü sensörler\nengel mesafesini gösteriyor.', 6),
    ('death_005000.pt', '5.000 ÖLÜM', 'Geçerli rota kısaldıkça ödül geliyor;\ndolanmayı öğreniyor.', 6),
    ('death_100000.pt', '100.000 ÖLÜM', 'Ortak politika daha tutarlı\nengel etrafı rota seçiyor.', 6),
)


def center(surface, font, text, color, y):
    image = font.render(text, True, color)
    surface.blit(image, ((CANVAS[0] - image.get_width()) // 2, y))


def subtitle(surface, fonts, lines):
    pygame.draw.rect(surface, (5, 8, 16), (45, 1040, 630, 125), border_radius=18)
    for index, line in enumerate(lines.split('\n')):
        center(surface, fonts['body'], line, TEXT, 1060 + index * 38)


def save(surface, directory, number):
    pygame.image.save(surface, directory / f'frame_{number:05d}.png')


def intro(surface, fonts):
    surface.fill(BACKGROUND)
    center(surface, fonts['title'], '3 ORGANİZMA • 6 YEM', TEXT, 230)
    center(surface, fonts['title'], '2 SABİT ENGEL', GREEN, 300)
    pygame.draw.rect(surface, OBSTACLE, (150, 465, 55, 220), border_radius=7)
    pygame.draw.rect(surface, OBSTACLE, (350, 590, 245, 55), border_radius=7)
    for x, color in zip((210, 365, 520), ORGANISM_COLORS):
        pygame.draw.circle(surface, color, (x, 540), 25)
    center(surface, fonts['small'], 'Bir beyin • üç ayrı hareket eden organizma', MUTED, 760)
    subtitle(surface, fonts, 'Üç organizma, altı yem\nve iki sabit engel.')


def outro(surface, fonts):
    surface.fill(BACKGROUND)
    center(surface, fonts['title'], 'NASIL ÖĞRENDİ?', TEXT, 245)
    center(surface, fonts['body'], 'Paylaşımlı Actor–Critic', GREEN, 410)
    center(surface, fonts['body'], '8 yönlü engel sensörü', TEXT, 500)
    center(surface, fonts['body'], 'Rota ilerlemesi ödülü', TEXT, 590)
    center(surface, fonts['small'], 'Yem +10 • çarpışma -1 • ölüm -10', MUTED, 750)
    subtitle(surface, fonts, 'Ödül, kuş uçuşu değil;\nengel etrafındaki geçerli rotaya göre.')


def render_segment(surface, fonts, spec, start_frame, frames_dir, run_dir, scenario, obstacles):
    checkpoint, title, words, seconds = spec
    agent = Agent(input_dim=18, output_dim=4, device=torch.device('cpu'), log_dir=None)
    agent.load_policy(run_dir / 'checkpoints' / checkpoint)
    world = pygame.Surface(WORLD)
    organisms = [Organism(world, *WORLD, agent=agent, color=color) for color in ORGANISM_COLORS]
    starts = ((130, 150), (390, 270), (670, 470))
    for organism, position in zip(organisms, starts):
        organism.x, organism.y, organism.energy, organism.score = *position, 100, 0
    foods = create_foods(world, organisms, scenario, obstacles)
    fields = create_fields(foods, obstacles)
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    actions, repeats = [0] * 3, [0] * 3
    trails = [[(item.x, item.y)] for item in organisms]
    labels, collisions = [], [0, 0, 0]
    for frame in range(seconds * FPS):
        for index, organism in enumerate(organisms):
            if repeats[index] == 0:
                state = organism.sense_environment(targets[organism], organisms, obstacle_distances(organism, obstacles))
                actions[index], _, _, _ = agent.select_action(state, deterministic=True)
                repeats[index] = scenario.action_repeat
            penalty = move_with_obstacles(organism, actions[index], obstacles)
            repeats[index] -= 1
            if penalty:
                collisions[index] = 12
            trails[index] = (trails[index] + [(organism.x, organism.y)])[-36:]
            eaten = next((food for food in foods if organism.food_distance(food) <= organism.radius + food.radius), None)
            if eaten:
                labels.append([eaten.x, eaten.y, 24, organism.color])
                foods.remove(eaten); fields.pop(eaten)
                replacement = spawn_food(world, *WORLD, organism, scenario.minimum_food_distance, scenario.maximum_food_distance, obstacles)
                foods.append(replacement); fields[replacement] = None
                for other in organisms:
                    if targets[other] is eaten:
                        targets[other] = choose_target(other, foods)
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                organism.energy += 10; organism.score += 10; repeats[index] = 0
            if organism.energy <= 0:
                organism.x, organism.y, organism.energy, organism.score = *starts[index], 100, 0
                targets[organism] = choose_target(organism, foods); repeats[index] = 0

        surface.fill(BACKGROUND)
        center(surface, fonts['title'], title, TEXT, 68)
        center(surface, fonts['subtitle'], 'Aynı harita • aynı başlangıç • eğitim aşaması değişiyor', MUTED, 138)
        pygame.draw.rect(surface, PANEL, WORLD_RECT, border_radius=18)
        world.fill((245, 248, 255))
        for rect in obstacles:
            pygame.draw.rect(world, OBSTACLE, rect, border_radius=6)
        for path, color in zip(trails, ORGANISM_COLORS):
            if len(path) > 1: pygame.draw.lines(world, color, False, path, 3)
        for food in foods: food.draw()
        for organism in organisms:
            target = targets[organism]
            pygame.draw.circle(world, organism.color, (int(target.x), int(target.y)), target.radius + 7, 2)
        for index, organism in enumerate(organisms):
            if collisions[index]: pygame.draw.circle(world, (240, 78, 78), (int(organism.x), int(organism.y)), organism.radius + 10, 3)
            organism.draw(); collisions[index] = max(0, collisions[index] - 1)
        for item in labels[:]:
            x, y, ttl, color = item; world.blit(fonts['world'].render('+10', True, color), (x - 16, y - 42 + (24 - ttl) * -2)); item[2] -= 1
            if item[2] <= 0: labels.remove(item)
        surface.blit(pygame.transform.smoothscale(world, WORLD_RECT.size), WORLD_RECT.topleft)
        progress = (frame + 1) / (seconds * FPS)
        pygame.draw.rect(surface, (65, 78, 112), (90, 878, 540, 10), border_radius=5)
        pygame.draw.rect(surface, GREEN, (90, 878, int(540 * progress), 10), border_radius=5)
        scores = '  '.join(f'{index + 1}: {organism.score:.0f}' for index, organism in enumerate(organisms))
        center(surface, fonts['body'], f'Puanlar  {scores}', TEXT, 932)
        center(surface, fonts['small'], 'Halka: hedef yem • Kırmızı: çarpışma • İz: seçilen rota', MUTED, 980)
        subtitle(surface, fonts, words)
        save(surface, frames_dir, start_frame + frame)
    return start_frame + seconds * FPS


def main():
    parser = argparse.ArgumentParser(description='Render an obstacle-learning YouTube Short.')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('runs') / 'shorts' / 'obstacle_learning_short.mp4')
    args = parser.parse_args()
    scenario = load_scenario(args.run / 'scenario.json')
    obstacles = load_obstacles(args.run / 'scenario.json')
    random.seed(23); pygame.init()
    fonts = {'title': pygame.font.SysFont('DejaVu Sans', 40, bold=True), 'subtitle': pygame.font.SysFont('DejaVu Sans', 19), 'body': pygame.font.SysFont('DejaVu Sans', 24), 'small': pygame.font.SysFont('DejaVu Sans', 17), 'world': pygame.font.SysFont('DejaVu Sans', 24, bold=True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = args.output.parent / f'{args.output.stem}_frames'
    if frames.exists(): shutil.rmtree(frames)
    frames.mkdir()
    surface, number = pygame.Surface(CANVAS), 0
    for _ in range(3 * FPS): intro(surface, fonts); save(surface, frames, number); number += 1
    for checkpoint in CHECKPOINTS: number = render_segment(surface, fonts, checkpoint, number, frames, args.run, scenario, obstacles)
    for _ in range(4 * FPS): outro(surface, fonts); save(surface, frames, number); number += 1
    subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', str(frames / 'frame_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)], check=True)
    shutil.rmtree(frames); pygame.quit(); print(args.output)


if __name__ == '__main__':
    main()
