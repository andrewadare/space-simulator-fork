import pygame
import asyncio
import argparse
import cProfile
import importlib
import yaml

from modules.utils import pre_render_text
from modules.config import (
    SpaceConfig,
    SimConfig,
    DynamicTaskGenerationConfig,
    RenderingMode,
    RenderingOptions,
)
from modules.task import generate_tasks
from modules.agent import generate_agents


parser = argparse.ArgumentParser(
    description="SPACE (Swarm Planning And Control Evalution) Simulator"
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the configuration file (default: --config=config.yaml)",
)


# Main game loop
async def game_loop(config: SpaceConfig):

    sim_config: SimConfig = config.simulation
    task_gen: DynamicTaskGenerationConfig = config.tasks.dynamic_task_generation
    rend_opts: RenderingOptions = config.simulation.rendering_options

    # Simulation timestep in seconds
    sampling_time = 1.0 / sim_config.sampling_freq

    # Initialize agents with behavior trees, giving them the information of current tasks
    tasks = generate_tasks(config)
    agents = generate_agents(tasks, config.agents)

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

    # Pre-rendered text for performance improvement
    mission_completed_text = pre_render_text("MISSION COMPLETED", 72, (0, 0, 0))

    running = True
    clock = pygame.time.Clock()
    game_paused = False
    mission_completed = False

    # Initialize simulation time
    simulation_time = 0.0
    last_print_time = 0.0  # Variable to track the last time tasks_left was printed

    # Initialize dynamic task generation time
    generation_count = 0
    last_generation_time = 0.0

    background_color = (224, 224, 224)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    game_paused = not game_paused

        if (
            sim_config.max_simulation_time > 0
            and simulation_time > sim_config.max_simulation_time
        ):
            running = False

        if not game_paused and not mission_completed:
            # Run behavior trees for each agent without rendering
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
                    new_task_id_start = len(tasks)
                    new_tasks = generate_tasks(
                        config,
                        task_quantity=task_gen.tasks_per_generation,
                        task_id_start=new_task_id_start,
                    )
                    tasks.extend(new_tasks)
                    last_generation_time = simulation_time
                    generation_count += 1
                    if sim_config.rendering_mode != RenderingMode.Headless:
                        print(
                            f"[{simulation_time:.2f}] Added {task_gen.tasks_per_generation} new tasks: Generation {generation_count}."
                        )

            # Rendering
            if sim_config.rendering_mode == RenderingMode.Screen:
                screen.fill(background_color)

                # Draw agents network topology
                if rend_opts.agent_communication_topology:
                    for agent in agents:
                        agent.draw_communication_topology(screen, agents)

                # Draw agents
                for agent in agents:
                    if rend_opts.agent_path_to_assigned_tasks:
                        agent.draw_path_to_assigned_tasks(screen)
                    if rend_opts.agent_tail:
                        agent.draw_tail(screen)
                    if rend_opts.agent_id:
                        agent.draw_agent_id(screen, font)
                    if rend_opts.agent_assigned_task_id:
                        agent.draw_assigned_task_id(screen, font)
                    if rend_opts.agent_work_done:
                        agent.draw_work_done(screen, font)
                    if rend_opts.agent_situation_awareness_circle:
                        agent.draw_situation_awareness_circle(screen)
                    agent.draw(screen)

                # Draw tasks with task_id displayed
                for task in tasks:
                    task.draw(screen)
                    if rend_opts.task_id:
                        task.draw_task_id(screen)

                # Display task quantity and elapsed simulation time
                task_time_text = pre_render_text(
                    f"Tasks left: {tasks_left}; Time: {simulation_time:.2f}s",
                    36,
                    (0, 0, 0),
                )
                screen.blit(task_time_text, (sim_config.screen_width - 350, 20))

                # Call draw_decision_making_status from the imported module if it exists
                # if hasattr(decision_making_module, "draw_decision_making_status"):
                #     decision_making_module.draw_decision_making_status(screen, agent)

                # Check if all tasks are completed
                if mission_completed:
                    text_rect = mission_completed_text.get_rect(
                        center=(
                            sim_config.screen_width // 2,
                            sim_config.screen_height // 2,
                        )
                    )
                    screen.blit(mission_completed_text, text_rect)

                pygame.display.flip()
                clock.tick(sim_config.sampling_freq * sim_config.speed_up_factor)

            elif sim_config.rendering_mode == "Terminal":
                print(f"[{simulation_time:.2f}] Tasks left: {tasks_left}")
                if simulation_time - last_print_time > 0.5:
                    last_print_time = simulation_time

                if mission_completed:
                    print(f"MISSION COMPLETED")
                    running = False
            else:  # None
                if mission_completed:
                    print(f"[{simulation_time:.2f}] MISSION COMPLETED")
                    running = False

    pygame.quit()


def main(config: SpaceConfig):
    asyncio.run(game_loop(config))


# Run the game
if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = SpaceConfig(**config_dict)

    if config.simulation.profiling_mode:
        cProfile.run("main(config)", sort="cumulative")
    else:
        main(config)
