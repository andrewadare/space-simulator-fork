import math

import pygame
import matplotlib.cm as cm

from modules.configuration_models import TaskConfig
from modules.task import Task


TASK_COLORS: dict[int, tuple[int, int, int]] = {}


def set_task_colors(task_config: TaskConfig) -> None:
    global TASK_COLORS

    # Generate task_colors based on tasks.quantity
    def generate_task_colors(quantity):
        # 'tab20' is a colormap with 20 distinct colors
        colors = cm.get_cmap("tab20", quantity)
        task_colors = {}
        for i in range(quantity):
            color = colors(i)  # Get color from colormap
            task_colors[i] = (
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            )
        return task_colors

    task_gen = task_config.dynamic_task_generation
    max_generations = task_gen.max_generations if task_gen.enabled else 0
    tasks_per_generation = task_gen.tasks_per_generation if task_gen.enabled else 0

    TASK_COLORS = generate_task_colors(
        task_config.quantity + tasks_per_generation * max_generations
    )


def draw_task(task: Task, screen):
    task.color = TASK_COLORS.get(task.task_id, (0, 0, 0))
    task.radius = task.amount / task.config.simulation.task_visualisation_factor
    if not task.completed:
        pygame.draw.circle(screen, task.color, task.position, int(task.radius))


def draw_task_id(task, screen):
    if not task.completed:
        font = pygame.font.Font(None, 15)
        text_surface = font.render(
            f"task_id {task.task_id}: {task.amount:.2f}", True, (250, 250, 250)
        )
        screen.blit(text_surface, (task.position[0], task.position[1]))


def draw_agent(agent, screen):
    size = 10
    angle = agent.rotation

    # Calculate the triangle points based on the current position and angle
    p1 = pygame.Vector2(
        agent.position.x + size * math.cos(angle),
        agent.position.y + size * math.sin(angle),
    )
    p2 = pygame.Vector2(
        agent.position.x + size * math.cos(angle + 2.5),
        agent.position.y + size * math.sin(angle + 2.5),
    )
    p3 = pygame.Vector2(
        agent.position.x + size * math.cos(angle - 2.5),
        agent.position.y + size * math.sin(angle - 2.5),
    )

    agent.color = TASK_COLORS.get(agent.assigned_task_id, (20, 20, 20))

    pygame.draw.polygon(screen, agent.color, [p1, p2, p3])


def draw_tail(agent, screen):
    # Draw track
    if len(agent.memory_location) >= 2:
        pygame.draw.lines(screen, agent.color, False, agent.memory_location, 1)


def draw_communication_topology(agent, screen, agents):
    # Draw lines to neighbor agents
    for neighbor_agent in agent.agents_nearby:
        if neighbor_agent.agent_id > agent.agent_id:
            neighbor_position = agents[neighbor_agent.agent_id].position
            pygame.draw.line(
                screen,
                (200, 200, 200),
                (int(agent.position.x), int(agent.position.y)),
                (int(neighbor_position.x), int(neighbor_position.y)),
            )


def draw_agent_id(agent, screen, font):
    # Draw assigned_task_id next to agent position
    text_surface = font.render(f"agent_id: {agent.agent_id}", True, (50, 50, 50))
    screen.blit(text_surface, (agent.position[0] + 10, agent.position[1] - 10))


def draw_assigned_task_id(agent, screen, font):
    # Draw assigned_task_id next to agent position
    if len(agent.planned_tasks) > 0:
        assigned_task_id_list = [task.task_id for task in agent.planned_tasks]
    else:
        assigned_task_id_list = agent.assigned_task_id
    text_surface = font.render(f"task_id: {assigned_task_id_list}", True, (50, 50, 50))
    screen.blit(text_surface, (agent.position[0] + 10, agent.position[1]))


def draw_work_done(agent, screen, font):
    # Draw assigned_task_id next to agent position
    text_surface = font.render(f"dist: {agent.distance_moved:.1f}", True, (50, 50, 50))
    screen.blit(text_surface, (agent.position[0] + 10, agent.position[1] + 10))
    text_surface = font.render(
        f"work: {agent.task_amount_done:.1f}", True, (50, 50, 50)
    )
    screen.blit(text_surface, (agent.position[0] + 10, agent.position[1] + 20))


def draw_situation_awareness_circle(agent, screen):
    # Draw the situation awareness radius circle
    if agent.situation_awareness_radius > 0:
        pygame.draw.circle(
            screen,
            agent.color,
            (agent.position[0], agent.position[1]),
            agent.situation_awareness_radius,
            1,
        )


def draw_path_to_assigned_tasks(agent, screen):
    # Starting position is the agent's current position
    start_pos = agent.position

    # Define line thickness
    line_thickness = 3  # Set the desired thickness for the lines
    # line_thickness = 16-4*agent.agent_id  # Set the desired thickness for the lines

    # For Debug
    color_list = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203),  # Pink
    ]

    # Iterate over the assigned tasks and draw lines connecting them
    for task in agent.planned_tasks:
        task_position = task.position
        pygame.draw.line(
            screen,
            # (255, 0, 0),  # Color for the path line (Red)
            color_list[agent.agent_id % len(color_list)],
            (int(start_pos.x), int(start_pos.y)),
            (int(task_position.x), int(task_position.y)),
            line_thickness,  # Thickness of the line
        )
        # Update the start position for the next segment
        start_pos = task_position
