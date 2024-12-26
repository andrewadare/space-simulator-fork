#!/usr/bin/env python

import pygame
import asyncio
import argparse
import cProfile
import yaml
from enum import Enum

import modules.visualization as vis
from modules.configuration_models import (
    SpaceConfig,
    SimConfig,
    RenderingMode,
    RenderingOptions,
)
from modules.factories import generate_tasks, generate_agents, DynamicTaskGenerator


parser = argparse.ArgumentParser(
    description="SPACE (Swarm Planning And Control Evalution) Simulator"
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the configuration file (default: --config=config.yaml)",
)
parser.add_argument(
    "--strategy",
    type=str,
    choices=["CBAA", "CBBA", "GRAPE", "FirstClaimGreedy"],
    default="CBBA",
    help="Task allocation strategy",
)


class LoopStatus(Enum):
    Running = 0
    Paused = 1
    Done = 2


def handle_pygame_events(status: LoopStatus) -> LoopStatus:
    """Update loop status from pygame events"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            status = LoopStatus.Done
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                status = LoopStatus.Done
            elif event.key == pygame.K_p:
                if status == LoopStatus.Running:
                    status = LoopStatus.Paused
                elif status == LoopStatus.Paused:
                    status = LoopStatus.Running
    return status


async def game_loop(config: SpaceConfig, strategy: str):

    sim_config: SimConfig = config.simulation
    rend_opts: RenderingOptions = config.simulation.rendering_options

    vis.set_task_colors(config.tasks)

    timestep = 1.0 / sim_config.sampling_freq  # seconds

    tasks = generate_tasks(config, config.tasks.quantity, 0)
    agents = generate_agents(tasks, config, strategy)
    task_generator = DynamicTaskGenerator(config.tasks.dynamic_task_generation)

    pygame.init()
    if sim_config.rendering_mode == RenderingMode.Screen:
        screen = pygame.display.set_mode(
            (sim_config.screen_width, sim_config.screen_height), pygame.RESIZABLE
        )
    else:
        screen = None  # No screen initialization if rendering is disabled

    pygame.display.set_caption(
        "SPACE (Swarm Planning And Control Evaluation) Simulator"
    )
    font = pygame.font.Font(None, 15)
    clock = pygame.time.Clock()
    status = LoopStatus.Running
    mission_completed: bool = False

    # Initialize simulation time
    simulation_time = 0.0
    last_print_time = 0.0  # Variable to track the last time tasks_left was printed

    background_color = (224, 224, 224)

    while status != LoopStatus.Done:

        status = handle_pygame_events(status)

        if simulation_time > sim_config.max_simulation_time:
            print("Simulation timed out.")
            status = LoopStatus.Done

        if status != LoopStatus.Running:
            continue

        for agent in agents:
            await agent.run_tree()
            agent.update(timestep)

        simulation_time += timestep
        tasks_left = sum(1 for task in tasks if not task.completed)
        if tasks_left == 0 and task_generator.done():
            mission_completed = True

        task_generator.update(simulation_time, tasks)

        # Rendering
        if sim_config.rendering_mode == RenderingMode.Screen:
            screen.fill(background_color)
            vis.draw_agents(agents, screen, rend_opts, font)
            vis.draw_tasks(
                tasks, screen, rend_opts, sim_config.task_visualisation_factor
            )
            vis.draw_task_status(tasks_left, simulation_time, screen)

            if mission_completed:
                vis.draw_mission_completed(screen)
                status = LoopStatus.Paused

            pygame.display.flip()
            clock.tick(sim_config.sampling_freq * sim_config.speed_up_factor)

        elif sim_config.rendering_mode == "Terminal":
            print(f"[{simulation_time:.2f}] Tasks left: {tasks_left}")
            if simulation_time - last_print_time > 0.5:
                last_print_time = simulation_time

            if mission_completed:
                print(f"MISSION COMPLETED")
                status = LoopStatus.Done
        else:
            if mission_completed:
                print(f"[{simulation_time:.2f}] MISSION COMPLETED")
                status = LoopStatus.Done

    pygame.quit()


def main(config: SpaceConfig, strategy: str):
    asyncio.run(game_loop(config, strategy))


# Run the game
if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = SpaceConfig(**config_dict)

    if config.simulation.profiling_mode:
        cProfile.run(f"main(config, {args.strategy})", sort="cumulative")
    else:
        main(config, args.strategy)
