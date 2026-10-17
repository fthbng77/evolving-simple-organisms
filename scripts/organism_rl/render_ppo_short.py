import argparse
import random
import shutil
import subprocess
from pathlib import Path

import pygame

from multi_main import ORGANISM_COLORS, choose_target, recolor_foods
from obstacles import move_with_obstacles, obstacle_distances, spawn_food
from organism import Organism
from ppo_agent import PPOAgent
from ppo_curriculum_main import HEIGHT, WIDTH, load_curriculum, rectangles


FPS = 30
CANVAS = (720, 1280)
WORLD_RECT = pygame.Rect(45, 335, 630, 472)
BACKGROUND, TEXT, MUTED = (246, 248, 247), (31, 42, 48), (91, 106, 113)
OBSTACLE, FOOD = (105, 116, 124), (232, 202, 0)
CHECKPOINTS = (
    ('episode_040000.pt', '40.000 BÖLÜM', 'İki engel arasında rota\nhâlâ kararsız.', 6),
    ('episode_060000.pt', '60.000 BÖLÜM', 'Hücreler daha fazla yem buluyor;\nçarpışmalar sürüyor.', 6),
    ('episode_085000.pt', '85.000 BÖLÜM', 'PPO daha çok yem topluyor;\nama her rota güvenilir değil.', 7),
)


def center(surface, font, text, color, top):
    image = font.render(text, True, color)
    surface.blit(image, ((CANVAS[0] - image.get_width()) // 2, top))


def subtitle(surface, fonts, lines):
    pygame.draw.rect(surface, (255, 255, 255), (42, 1030, 636, 120), border_radius=14)
    pygame.draw.rect(surface, (216, 224, 226), (42, 1030, 636, 120), 1, border_radius=14)
    for index, line in enumerate(lines.split('\n')):
        center(surface, fonts['body'], line, TEXT, 1055 + index * 35)


def write_frame(surface, frames, index):
    pygame.image.save(surface, frames / f'frame_{index:05d}.png')


def reset_episode(world, agent, curriculum, obstacles, seed):
    random.seed(seed)
    organisms = [Organism(world, WIDTH, HEIGHT, agent=agent, color=color) for color in ORGANISM_COLORS]
    foods = [
        spawn_food(world, WIDTH, HEIGHT, organisms[index % len(organisms)], curriculum['food_distance']['minimum'], curriculum['food_distance']['maximum'], obstacles)
        for index in range(curriculum['food_count'])
    ]
    targets = {organism: choose_target(organism, foods) for organism in organisms}
    recolor_foods(foods, targets)
    return organisms, foods, targets


def draw_world(world, organisms, foods, targets, obstacles, trails, collisions):
    world.fill(BACKGROUND)
    pygame.draw.rect(world, (225, 231, 233), (0, 0, WIDTH, HEIGHT), 3)
    for obstacle in obstacles:
        pygame.draw.rect(world, OBSTACLE, obstacle, border_radius=6)
        pygame.draw.rect(world, (69, 79, 87), obstacle, 2, border_radius=6)
    for path, color in zip(trails, ORGANISM_COLORS):
        if len(path) > 1:
            pygame.draw.lines(world, color, False, path, 2)
    for food in foods:
        food.draw()
    for index, organism in enumerate(organisms):
        target = targets[organism]
        pygame.draw.circle(world, organism.color, (int(target.x), int(target.y)), target.radius + 7, 2)
        if collisions[index]:
            pygame.draw.circle(world, (220, 88, 80), (int(organism.x), int(organism.y)), organism.radius + 8, 2)
        organism.draw()


def render_segment(surface, fonts, spec, start, frames, run, curriculum, obstacles):
    file_name, title, words, seconds = spec
    agent = PPOAgent(18, 4, log_dir=None)
    agent.load_policy(run / 'checkpoints' / file_name)
    world = pygame.Surface((WIDTH, HEIGHT))
    organisms, foods, targets = reset_episode(world, agent, curriculum, obstacles, 100_000)
    actions, repeats, scores, collisions = [0] * 3, [0] * 3, [0] * 3, [0] * 3
    trails = [[(organism.x, organism.y)] for organism in organisms]
    for frame in range(seconds * FPS):
        for index, organism in enumerate(organisms):
            if repeats[index] == 0:
                state = organism.sense_environment(targets[organism], organisms, obstacle_distances(organism, obstacles))
                actions[index] = agent.select_action(state, deterministic=True)
                repeats[index] = curriculum['action_repeat']
            penalty = move_with_obstacles(organism, actions[index], obstacles)
            collisions[index] = 10 if penalty else max(0, collisions[index] - 1)
            repeats[index] -= 1
            trails[index] = (trails[index] + [(organism.x, organism.y)])[-48:]
            eaten = next((food for food in foods if organism.food_distance(food) <= organism.radius + food.radius), None)
            if eaten:
                foods.remove(eaten)
                new_food = spawn_food(world, WIDTH, HEIGHT, organism, curriculum['food_distance']['minimum'], curriculum['food_distance']['maximum'], obstacles)
                foods.append(new_food)
                for other in organisms:
                    if targets[other] is eaten:
                        targets[other] = choose_target(other, foods)
                targets[organism] = choose_target(organism, foods)
                recolor_foods(foods, targets)
                organism.energy += 10
                scores[index] += 1
        draw_world(world, organisms, foods, targets, obstacles, trails, collisions)
        surface.fill(BACKGROUND)
        center(surface, fonts['title'], title, TEXT, 74)
        center(surface, fonts['small'], '3 hücre • 6 yem • 2 sabit engel', MUTED, 137)
        pygame.draw.rect(surface, (255, 255, 255), WORLD_RECT.inflate(12, 12), border_radius=14)
        surface.blit(pygame.transform.smoothscale(world, WORLD_RECT.size), WORLD_RECT.topleft)
        progress = (frame + 1) / (seconds * FPS)
        pygame.draw.rect(surface, (215, 222, 224), (90, 870, 540, 9), border_radius=5)
        pygame.draw.rect(surface, (102, 201, 135), (90, 870, int(540 * progress), 9), border_radius=5)
        center(surface, fonts['body'], f"Yem: {'  /  '.join(map(str, scores))}", TEXT, 923)
        center(surface, fonts['small'], 'İz: hareket • Halka: aktif hedef • Kırmızı: çarpışma', MUTED, 970)
        subtitle(surface, fonts, words)
        write_frame(surface, frames, start + frame)
    return start + seconds * FPS


def intro(surface, fonts):
    surface.fill(BACKGROUND)
    center(surface, fonts['title'], 'ENGELLERİ AŞABİLİR Mİ?', TEXT, 250)
    center(surface, fonts['body'], '3 hücre • 6 yem • PPO', (255, 128, 110), 350)
    center(surface, fonts['small'], 'Hücre önce yem bulmayı, sonra rota seçmeyi öğreniyor.', MUTED, 470)
    subtitle(surface, fonts, 'İki engel, aynı anda\nhedef seçmeyi zorlaştırıyor.')


def outro(surface, fonts):
    surface.fill(BACKGROUND)
    center(surface, fonts['title'], 'SONUÇ', TEXT, 245)
    center(surface, fonts['body'], 'Daha çok yem • ama kararsız rota', (255, 128, 110), 365)
    center(surface, fonts['small'], 'Sonraki adım: daha geniş geçitler ve çarpışma cezası.', MUTED, 475)
    subtitle(surface, fonts, 'Yöntem: PPO + aşamalı eğitim.\nHer aşama kör testle ölçülüyor.')


def main():
    parser = argparse.ArgumentParser(description='Render a PPO obstacle-learning vertical Short.')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/shorts/ppo_obstacle_learning_short.mp4'))
    args = parser.parse_args()
    curriculum = load_curriculum(args.run / 'scenario.json')
    obstacles = rectangles(curriculum['stages'][-1])
    pygame.init()
    fonts = {'title': pygame.font.SysFont('DejaVu Sans', 38, bold=True), 'body': pygame.font.SysFont('DejaVu Sans', 23), 'small': pygame.font.SysFont('DejaVu Sans', 17)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = args.output.parent / f'{args.output.stem}_frames'
    if frames.exists(): shutil.rmtree(frames)
    frames.mkdir()
    surface, index = pygame.Surface(CANVAS), 0
    for _ in range(3 * FPS): intro(surface, fonts); write_frame(surface, frames, index); index += 1
    for spec in CHECKPOINTS: index = render_segment(surface, fonts, spec, index, frames, args.run, curriculum, obstacles)
    for _ in range(4 * FPS): outro(surface, fonts); write_frame(surface, frames, index); index += 1
    subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', str(frames / 'frame_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)], check=True)
    shutil.rmtree(frames)
    pygame.quit()


if __name__ == '__main__':
    main()
