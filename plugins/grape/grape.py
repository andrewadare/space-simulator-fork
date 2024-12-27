import random
import copy

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
)

from modules.utils import pre_render_text
from modules.agent import Agent
from modules.task import Task
from modules.configuration_models import AgentConfig


class GRAPEConfig(BaseModel):
    execute_movements_during_convergence: bool = Field(default=False)
    cost_weight_factor: PositiveFloat
    social_inhibition_factor: PositiveFloat
    initialize_partition: str  # Options: ""; "Distance"
    reinitialize_partition_on_completion: str  # Options: ""; "Distance";


class GRAPE:
    def __init__(self, agent_id: int, config: GRAPEConfig, agent_config: AgentConfig):
        self.agent_id = agent_id
        self.satisfied = False
        self.evolution_number = 0
        self.time_stamp = 0
        self.assigned_task = None
        self.config = config
        self.agent_config = agent_config
        self.partition: dict[int, set] = dict()
        self.partition_initialized = False

        self.current_utilities = {}
        self.message_to_share = {  # Message Initialization
            "agent_id": self.agent_id,
            "partition": self.partition,
            "evolution_number": self.evolution_number,
            "time_stamp": self.time_stamp,
        }

        # Since assigned task is queried several times,
        # use task_id => Task map for fast lookup.
        self.task_map: dict[int, Task] = dict()

    def initialize_partition(self, agents: list[Agent], tasks: list[Task]):
        self.partition = {task.task_id: set() for task in tasks}

        if self.config.initialize_partition == "Distance":
            if tasks and agents:
                self.partition = self.initialize_partition_by_distance(
                    agents, tasks, self.partition
                )
                self.assigned_task = self.get_assigned_task_from_partition(
                    self.partition
                )
        self.partition_initialized = True

    def initialize_partition_by_distance(self, agents_info, tasks_info, partition):
        for agent in agents_info:
            task_distance = {
                task.task_id: (
                    float("inf")
                    if task.completed
                    else np.linalg.norm(agent.position - task.position)
                )
                for task in tasks_info
            }
            if len(task_distance) > 0:
                preferred_task_id = min(task_distance, key=task_distance.get)

                # Ensure the task_id key exists in the partition.
                # Set tis value as empty set if it doesn't already exist (This is for dynamic task generation)
                self.partition.setdefault(preferred_task_id, set())
                partition[preferred_task_id].add(agent.agent_id)
        return partition

    def get_neighbor_agents_info_in_partition(
        self, partition, nearby_agents: list[Agent]
    ):
        _neighbor_agents_info = [
            neighbor_agent
            for neighbor_agent in nearby_agents
            if neighbor_agent.agent_id in partition[self.assigned_task.task_id]
        ]
        return _neighbor_agents_info

    def decide(self, blackboard: dict, agent_position: np.ndarray):
        """
        Output:
            - `task_id`, if task allocation works well
            - `None`, otherwise
        """

        _local_tasks_info = blackboard["local_tasks_info"]
        _local_agents_info = blackboard["local_agents_info"]

        for task in blackboard["local_tasks_info"]:
            self.task_map[task.task_id] = task

        if not self.partition_initialized:
            self.initialize_partition(_local_agents_info, _local_tasks_info)

        # Check if the existing task is done
        if self.assigned_task is not None and self.assigned_task.completed:
            _neighbor_agents_info = self.get_neighbor_agents_info_in_partition(
                self.partition, _local_agents_info
            )

            # Default routine
            # Empty the previous task's coalition
            self.partition[self.assigned_task.task_id] = set()
            self.assigned_task = None
            self.satisfied = False

            # Special routine
            if self.config.reinitialize_partition_on_completion == "Distance":
                self.partition = self.initialize_partition_by_distance(
                    _neighbor_agents_info, _local_tasks_info, self.partition
                )
                self.assigned_task = self.get_assigned_task_from_partition(
                    self.partition
                )

        # Give up the decision-making process if there is no task nearby
        if len(_local_tasks_info) == 0:
            return None

        # GRAPE algorithm for each agent (Phase 1)
        if len(_local_tasks_info) == 0:
            return None
        if not self.satisfied:
            _max_task_id, _max_utility = self.find_max_utility_task(
                _local_tasks_info, agent_position
            )
            self.assigned_task = self.get_assigned_task_from_partition(self.partition)
            if _max_utility > self.compute_utility(self.assigned_task, agent_position):
                self.update_partition(_max_task_id)
                self.evolution_number += 1
                self.time_stamp = random.uniform(0, 1)

            self.satisfied = True

            # Broadcasting # NOTE: Implemented separately
            self.message_to_share = {
                "agent_id": self.agent_id,
                "partition": self.partition,
                "evolution_number": self.evolution_number,
                "time_stamp": self.time_stamp,
            }

            return None

        # D-Mutex (Phase 2)
        self.evolution_number, self.time_stamp, self.partition, self.satisfied = (
            self.distributed_mutex(blackboard["messages_received"])
        )

        self.assigned_task = self.get_assigned_task_from_partition(self.partition)

        if not self.satisfied:
            if not self.config.execute_movements_during_convergence:
                # Neutralise the agent's current movement during converging to a Nash stable partition
                blackboard["stop_moving"] = True

        return (
            copy.deepcopy(self.assigned_task.task_id)
            if self.assigned_task is not None
            else None
        )

    def discard_myself_from_coalition(self, task):
        if task is not None:
            self.partition[task.task_id].discard(self.agent_id)

    def update_partition(self, preferred_task_id):
        self.discard_myself_from_coalition(self.assigned_task)
        self.partition[preferred_task_id].add(self.agent_id)

    def find_max_utility_task(self, tasks_info, agent_position: np.ndarray):
        _current_utilities = {
            task.task_id: (
                self.compute_utility(task, agent_position)
                if not task.completed
                else float("-inf")
            )
            for task in tasks_info
        }

        _max_task_id = max(_current_utilities, key=_current_utilities.get)
        _max_utility = _current_utilities[_max_task_id]

        self.current_utilities = _current_utilities

        return _max_task_id, _max_utility

    def compute_utility(self, task: Task, agent_position: np.ndarray) -> float:
        # Individual Utility Function
        if task is None:
            return float("-inf")

        # Ensure the task_id key exists in the partition.
        # Set tis value as empty set if it doesn't already exist (This is for dynamic task generation)
        self.partition.setdefault(task.task_id, set())
        num_collaborator = len(self.partition[task.task_id])
        if self.agent_id not in self.partition[task.task_id]:
            num_collaborator += 1

        distance = np.linalg.norm(agent_position - task.position)
        utility = task.amount / (
            num_collaborator
        ) - self.config.cost_weight_factor * distance * (
            num_collaborator**self.config.social_inhibition_factor
        )
        return utility

    def distributed_mutex(self, messages_received):
        _satisfied = True
        _evolution_number = self.evolution_number
        _partition = self.partition
        _time_stamp = self.time_stamp

        for message in messages_received:
            if message["evolution_number"] > _evolution_number or (
                message["evolution_number"] == _evolution_number
                and message["time_stamp"] > _time_stamp
            ):
                _evolution_number = message["evolution_number"]
                _time_stamp = message["time_stamp"]
                _partition = message["partition"]

                _satisfied = False

        _final_partition = {k: v.copy() for k, v in _partition.items()}
        return _evolution_number, _time_stamp, _final_partition, _satisfied

    def get_assigned_task_id(self, partition: dict) -> int:
        for task_id, coalition_members_id in partition.items():
            if self.agent_id in coalition_members_id:
                return task_id
        return -1

    def get_assigned_task_from_partition(self, partition) -> int | None:
        id = self.get_assigned_task_id(partition)
        return self.task_map.get(id)


def draw_decision_making_status(screen, agent):
    if "evolution_number" in agent.message_to_share:  # For GRAPE
        partition_evolution_number = agent.message_to_share["evolution_number"]
        partition_evolution_number_text = pre_render_text(
            f"Partition evolution number: {partition_evolution_number}", 36, (0, 0, 0)
        )
        screen.blit(partition_evolution_number_text, (20, 20))
