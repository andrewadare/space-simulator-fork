import random

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from modules.configuration_models import AgentConfig
from modules.task import Task


class FirstClaimGreedyConfig(BaseModel):
    mode: str
    weight_factor_cost: PositiveFloat
    enforced_collaboration: bool = Field(default=False)


class FirstClaimGreedy:
    """Task selection within each agent's `situation_awareness_radius`"""

    def __init__(
        self, agent, config: FirstClaimGreedyConfig, agent_config: AgentConfig
    ):
        self.agent_id = agent.agent_id
        self.config = config
        self.agent_config = agent_config
        self.assigned_task = None

    def decide(self, blackboard: dict, agent_position: np.ndarray):
        """
        Output:
            - `task_id`, if task allocation works well
            - `None`, otherwise
        """
        local_tasks: list[Task] = blackboard["local_tasks_info"]

        # Check if the existing task is done
        if self.assigned_task is not None and self.assigned_task.completed:
            self.assigned_task = None

        # Give up the decision-making process if there is no task nearby
        if len(local_tasks) == 0:
            self.assigned_task = None
            self.message_to_share = {
                "agent_id": self.agent_id,
                "assigned_task_id": None,
            }
            return None

        # Given that there is only one task nearby, then enforced to select this
        if self.config.enforced_collaboration and len(local_tasks) == 1:
            self.assigned_task = local_tasks[0]
            return self.assigned_task.task_id

        # Look for a task within situation awareness radius if there is no existing assigned task
        if self.assigned_task is None:
            unassigned_tasks = self.filter_unassigned_tasks_from_neighbor_messages(
                local_tasks,
                blackboard["messages_received"],
            )

            if len(unassigned_tasks) == 0:
                self.assigned_task = None
                self.message_to_share = {
                    "agent_id": self.agent_id,
                    "assigned_task_id": None,
                }
                return None

            # Choose a task randomly
            if self.config.mode == "Random":
                target_task_id = random.choice(unassigned_tasks).task_id

            # Choose the closest task
            elif self.config.mode == "MinDist":
                target_task_id = self.find_min_dist_task(
                    unassigned_tasks, agent_position
                )

            # Choose the task providing the maximum utility
            elif self.config.mode == "MaxUtil":
                target_task_id = self.find_max_utility_task(
                    unassigned_tasks, agent_position
                )

            else:
                raise ValueError(f"Unrecognized selection mode: {self.config.mode}")

            for task in local_tasks:
                if task.task_id == target_task_id:
                    self.assigned_task = task

            self.message_to_share = {
                "agent_id": self.agent_id,
                "assigned_task_id": self.assigned_task.task_id,
            }

        return self.assigned_task.task_id

    def filter_unassigned_tasks_from_neighbor_messages(
        self, tasks: list[Task], messages_received
    ):
        occupied_tasks_id = []
        for message in messages_received:
            occupied_tasks_id.append(message.get("assigned_task_id"))

        unassigned_tasks = [
            task for task in tasks if task.task_id not in occupied_tasks_id
        ]

        return unassigned_tasks

    def find_min_dist_task(self, tasks: list[Task], agent_position: np.ndarray):
        _tasks_distance = {
            task.task_id: (
                self.compute_distance(task, agent_position)
                if not task.completed
                else float("inf")
            )
            for task in tasks
        }
        _min_task_id = min(_tasks_distance, key=_tasks_distance.get)
        return _min_task_id

    def find_max_utility_task(self, tasks: list[Task]):
        _current_utilities = {
            task.task_id: (
                self.compute_utility(task) if not task.completed else float("-inf")
            )
            for task in tasks
        }

        _max_task_id = max(_current_utilities, key=_current_utilities.get)

        return _max_task_id

    def compute_utility(self, task: Task, agent_position: np.ndarray):
        if task is None:
            return float("-inf")

        distance = self.compute_distance(task.position, agent_position)
        return task.amount - self.config.weight_factor_cost * distance

    def compute_distance(self, task: Task, agent_position: np.ndarray):

        if task is None:
            return float("inf")

        distance = np.linalg.norm(agent_position - task.position)
        return distance
