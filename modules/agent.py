import numpy as np
import math
import random
from collections import deque

from modules.behavior_tree import Status, ReturnsStatus
from modules.configuration_models import AgentConfig, OperatingArea
from modules.task import Task


def create_random_point(bounds: OperatingArea) -> tuple[float, float]:
    x = random.uniform(bounds.x_min, bounds.x_max)
    y = random.uniform(bounds.y_min, bounds.y_max)
    return (x, y)


class Agent:
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        tasks: list[Task],
        bounds: OperatingArea,
        agent_config: AgentConfig,
    ):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.rotation = 0
        self.bounds = bounds
        self.params = agent_config
        self.tail = deque(maxlen=400)
        self.tasks_info: list[Task] = tasks  # TODO see README
        self.all_agents: list[Agent] = []  # TODO see README
        self.message_to_share = {}
        self.assigned_task_id = None  # Local decision-making result.
        self.distance_moved = 0.0
        self.task_amount_done = 0.0

        # Shared with and modified by task allocator
        self.blackboard = dict(
            messages_received=[],
            # For visualization. Not used by all strategies.
            planned_tasks=[],
        )

        # Agent behaviors for self.tree
        self.node_callbacks: dict[str, ReturnsStatus] = dict(
            LocalSensingNode=self.sense,
            DecisionMakingNode=self.decide_task,
            TaskExecutingNode=self.goto_task,
            ExplorationNode=self.explore,
        )

        self.task_assigner = None
        self.tree = None

        # For self.explore callback
        self.exploration_time = 0.0
        self.random_waypoint = (0, 0)

    def sense(self) -> Status:
        """Find waypoints and other agents in this agent's vicinity."""
        self.blackboard["local_agents_info"] = self.get_agents_nearby()
        self.blackboard["local_tasks_info"] = self.get_tasks_nearby(
            incomplete_only=True
        )
        for other_agent in self.blackboard["local_agents_info"]:
            if other_agent.agent_id != self.agent_id:
                self.blackboard["messages_received"].append(
                    other_agent.message_to_share
                )
        return Status.SUCCESS

    def decide_task(self) -> Status:
        """Decide which waypoint to visit."""
        self.assigned_task_id = self.task_assigner.decide(
            self.blackboard, self.params.timestep
        )

        # "Publish" decision making info
        self.message_to_share = self.task_assigner.message_to_share

        # Clear inbox now that self.task_assigner is done reading agent's mail.
        self.blackboard["messages_received"] = []

        status = Status.FAILURE if self.assigned_task_id is None else Status.SUCCESS
        return status

    def goto_task(self) -> Status:
        """Go to assigned task position."""

        if self.assigned_task_id is not None:
            goal: Task = self.tasks_info[self.assigned_task_id]

            # Check if agent reached the task position.
            # NOTE: in original implementation, threshold_done_by_arrival
            # is added to task.radius. Here, use agent.radius for same effect.
            distance = np.linalg.norm(goal.position - self.position)
            if distance < goal.radius + self.params.radius:
                if goal.completed:
                    return Status.SUCCESS
                self.tasks_info[self.assigned_task_id].reduce_amount(
                    self.params.work_rate * self.params.timestep
                )
                self.task_amount_done += self.params.work_rate
            self.follow(goal.position)
        return Status.RUNNING

    def explore(self) -> Status:
        """Look busy by moving to a random imaginary waypoint."""
        if self.exploration_time > self.params.random_exploration_duration:
            self.random_waypoint = create_random_point(self.bounds)
            self.exploration_time = 0
        self.exploration_time += self.params.timestep
        self.follow(self.random_waypoint)
        return Status.RUNNING

    async def run_tree(self):
        return await self.tree.run()

    def follow(self, target: np.ndarray):
        offset = target - self.position
        distance = np.linalg.norm(offset)
        direction = offset / distance
        speed = self.params.max_speed
        if distance < self.params.target_approaching_radius:
            speed *= distance / self.params.target_approaching_radius

        self.acceleration = self.limit(
            self.acceleration + speed * direction - self.velocity, self.params.max_accel
        )

    def update(self, timestep: float):
        # Update velocity and position
        self.velocity += self.acceleration * timestep
        self.velocity = self.limit(self.velocity, self.params.max_speed)
        self.position += self.velocity * timestep
        self.acceleration *= 0  # Reset acceleration

        # Calculate the distance moved in this update and add to distance_moved
        self.distance_moved += np.linalg.norm(self.velocity) * timestep

        self.tail.append([*self.position])

        # Update rotation
        desired_rotation = math.atan2(self.velocity[1], self.velocity[0])
        rotation_diff = desired_rotation - self.rotation
        while rotation_diff > math.pi:
            rotation_diff -= 2 * math.pi
        while rotation_diff < -math.pi:
            rotation_diff += 2 * math.pi

        # Limit angular velocity
        if abs(rotation_diff) > self.params.max_angular_speed:
            rotation_diff = math.copysign(self.params.max_angular_speed, rotation_diff)

        self.rotation += rotation_diff * timestep

    def reset_movement(self):
        self.velocity *= 0
        self.acceleration *= 0

    def limit(self, vector: np.ndarray, max_value: float):
        mag = np.linalg.norm(vector)
        if mag > max_value:
            vector *= max_value / mag
        return vector

    def set_assigned_task_id(self, task_id):
        self.assigned_task_id = task_id

    def get_agents_nearby(self):
        r2 = self.params.communication_radius**2
        nearby_agents: list[Agent] = []
        for agent in self.all_agents:
            if agent.agent_id == self.agent_id:
                continue
            d = agent.position - self.position
            if d @ d <= r2:
                nearby_agents.append(agent)

        return nearby_agents

    def get_tasks_nearby(self, incomplete_only=False):
        if incomplete_only:
            tasks = [t for t in self.tasks_info if not t.completed]
        else:
            tasks = self.tasks_info

        r2 = self.params.situation_awareness_radius**2
        nearby_tasks: list[Task] = []
        for task in tasks:
            d = task.position - self.position
            if d @ d <= r2:
                nearby_tasks.append(task)

        return nearby_tasks
