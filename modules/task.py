import pygame
import random
from modules.utils import generate_positions, generate_task_colors
from modules.configuration_models import SpaceConfig


# Temporary gloabl until drawing code is refactored.
# This name is imported by agent.py
task_colors = None


class Task:
    def __init__(self, task_id, position, config: SpaceConfig):
        self.task_id = task_id
        self.position = pygame.Vector2(position)
        self.config = config
        self.amount = random.uniform(config.tasks.amounts.min, config.tasks.amounts.max)

        self.radius = self.amount / config.simulation.task_visualisation_factor
        self.completed = False

    def set_done(self):
        self.completed = True

    def reduce_amount(self, work_rate):
        sampling_time = 1.0 / self.config.simulation.sampling_freq
        self.amount -= work_rate * sampling_time
        if self.amount <= 0:
            self.set_done()

    def draw(self, screen):
        global task_colors
        if task_colors is None:
            task_gen = self.config.tasks.dynamic_task_generation
            max_generations = task_gen.max_generations if task_gen.enabled else 0
            tasks_per_generation = (
                task_gen.tasks_per_generation if task_gen.enabled else 0
            )
            task_colors = generate_task_colors(
                self.config.tasks.quantity + tasks_per_generation * max_generations
            )

        color = task_colors.get(self.task_id, (0, 0, 0))
        self.radius = self.amount / self.config.simulation.task_visualisation_factor
        if not self.completed:
            pygame.draw.circle(screen, color, self.position, int(self.radius))

    def draw_task_id(self, screen):
        if not self.completed:
            font = pygame.font.Font(None, 15)
            text_surface = font.render(
                f"task_id {self.task_id}: {self.amount:.2f}", True, (250, 250, 250)
            )
            screen.blit(text_surface, (self.position[0], self.position[1]))


def generate_tasks(
    config: SpaceConfig, task_quantity: int | None = None, task_id_start: int = 0
) -> list[Task]:
    if task_quantity is None:
        task_quantity = config.tasks.quantity

    tasks_positions = generate_positions(
        config.tasks.quantity,
        config.tasks.locations.x_min,
        config.tasks.locations.x_max,
        config.tasks.locations.y_min,
        config.tasks.locations.y_max,
        radius=config.tasks.locations.non_overlap_radius,
    )

    # Initialize tasks
    tasks = [
        Task(idx + task_id_start, pos, config)
        for idx, pos in enumerate(tasks_positions)
    ]
    return tasks
