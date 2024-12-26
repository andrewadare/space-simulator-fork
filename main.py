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
    DynamicTaskGenerationConfig,
    RenderingMode,
    RenderingOptions,
)
from modules.factories import generate_tasks, generate_agents
from modules.task import Task


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


def add_new_tasks(num_tasks: int, tasks: list[Task]):
    new_task_id_start = len(tasks)
    new_tasks = generate_tasks(config, num_tasks, new_task_id_start)
    tasks.extend(new_tasks)


# Main game loop
async def game_loop(config: SpaceConfig, strategy: str):

    sim_config: SimConfig = config.simulation
    task_gen: DynamicTaskGenerationConfig = config.tasks.dynamic_task_generation
    rend_opts: RenderingOptions = config.simulation.rendering_options

    vis.set_task_colors(config.tasks)

    # Simulation timestep in seconds
    sampling_time = 1.0 / sim_config.sampling_freq

    # Initialize agents with behavior trees, giving them the information of current tasks
    tasks = generate_tasks(config, config.tasks.quantity, 0)
    agents = generate_agents(tasks, config, strategy)

    # Initialize pygame
    pygame.init()
    if sim_config.rendering_mode == RenderingMode.Screen:
        screen = pygame.display.set_mode(
            (sim_config.screen_width, sim_config.screen_height), pygame.RESIZABLE
        )
    else:
        screen = None  # No screen initialization if rendering is disabled

    # Set logo and title
    logo_image_path = "assets/logo.jpg"  # Change to the path of your logo image
    logo = pygame.image.load(logo_image_path)
    pygame.display.set_icon(logo)
    pygame.display.set_caption("SPACE(Swarm Planning And Control Evaluation) Simulator")
    font = pygame.font.Font(None, 15)

    clock = pygame.time.Clock()
    status = LoopStatus.Running
    mission_completed = False

    # Initialize simulation time
    simulation_time = 0.0
    last_print_time = 0.0  # Variable to track the last time tasks_left was printed

    # Initialize dynamic task generation time
    generation_count = 0
    last_generation_time = 0.0

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
            agent.update(sampling_time)

        # Status retrieval
        simulation_time += sampling_time
        tasks_left = sum(1 for task in tasks if not task.completed)
        if tasks_left == 0:
            mission_completed = (
                not task_gen.enabled or generation_count == task_gen.max_generations
            )

        # Dynamic task generation
        if task_gen.enabled and generation_count < task_gen.max_generations:
            if simulation_time - last_generation_time >= task_gen.interval_seconds:
                add_new_tasks(task_gen.tasks_per_generation, tasks)
                last_generation_time = simulation_time
                generation_count += 1
                if sim_config.rendering_mode != RenderingMode.Headless:
                    print(
                        f"[{simulation_time:.2f}] Added {task_gen.tasks_per_generation} new tasks: "
                        f"Generation {generation_count}."
                    )

        # Rendering
        if sim_config.rendering_mode == RenderingMode.Screen:
            screen.fill(background_color)
            vis.draw_agents(agents, screen, rend_opts, font)
            vis.draw_tasks(
                tasks, screen, rend_opts, sim_config.task_visualisation_factor
            )
            vis.draw_task_status(tasks_left, simulation_time, screen)

            # Call draw_decision_making_status from the imported module if it exists
            # if hasattr(decision_making_module, "draw_decision_making_status"):
            #     decision_making_module.draw_decision_making_status(screen, agent)

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
