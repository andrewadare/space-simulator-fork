import numpy as np
import math
import random
from pathlib import Path
from collections import deque

from modules.behavior_tree import create_behavior_tree, Status, Node, ReturnsStatus
from modules.configuration_models import AgentConfig, OperatingArea
from modules.task import Task


def get_random_point(bounds: OperatingArea) -> tuple[float, float]:
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
        conf: AgentConfig,
    ):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.rotation = 0
        self.bounds = bounds
        self.params = conf
        self.tail = deque(maxlen=400)
        self.blackboard = {"messages_received": []}
        self.tasks_info: list[Task] = tasks  # TODO see README
        self.all_agents: list[Agent] = []  # TODO see README
        self.agents_nearby: list[Agent] = []
        self.message_to_share = {}
        self.assigned_task_id = None  # Local decision-making result.
        self.planned_tasks = []  # Local decision-making result.
        self.distance_moved = 0.0
        self.task_amount_done = 0.0

        # TODO: find a way to make this external to Agent
        # self.task_assigner = create_task_decider(self, conf.decision_making)
        self.task_assigner = None

        # Create behavior tree for this agent.
        # Agent behaviors are bound to action nodes as callbacks.
        self.node_callbacks: dict[str, ReturnsStatus] = dict(
            LocalSensingNode=self.sense,
            DecisionMakingNode=self.decide_task,
            TaskExecutingNode=self.goto_task,
            ExplorationNode=self.explore,
        )
        self.tree: Node = create_behavior_tree(
            Path("bt_xml") / conf.behavior_tree_xml, self.node_callbacks
        )

        # For self.explore callback
        self.exploration_time = 0.0
        self.random_waypoint = (0, 0)

    def sense(self) -> Status:
        """Find waypoints and other agents in this agent's vicinity."""
        self.blackboard["local_tasks_info"] = self.get_tasks_nearby(
            incomplete_only=True
        )
        self.blackboard["local_agents_info"] = self.local_message_receive()
        self.blackboard["LocalSensingNode"] = Status.SUCCESS
        return Status.SUCCESS

    def decide_task(self) -> Status:
        """Decide which waypoint to visit."""
        assigned_task_id = self.task_assigner.decide(
            self.blackboard, self.params.timestep
        )

        # "Publish" decision making info
        self.message_to_share = self.task_assigner.message_to_share

        status = Status.FAILURE if assigned_task_id is None else Status.SUCCESS
        self.set_assigned_task_id(assigned_task_id)
        self.blackboard["assigned_task_id"] = assigned_task_id
        self.blackboard["DecisionMakingNode"] = status
        return status

    def goto_task(self) -> Status:
        """Go to assigned task position."""
        assigned_task_id = self.blackboard.get("assigned_task_id")

        if assigned_task_id is not None:
            goal: Task = self.tasks_info[assigned_task_id]

            # Check if agent reached the task position.
            # NOTE: in original implementation, threshold_done_by_arrival
            # is added to task.radius. Here, use agent.radius for same effect.
            distance = np.linalg.norm(goal.position - self.position)
            if distance < goal.radius + self.params.radius:
                if goal.completed:
                    self.blackboard["TaskExecutingNode"] = Status.SUCCESS
                    return Status.SUCCESS
                self.tasks_info[assigned_task_id].reduce_amount(
                    self.params.work_rate * self.params.timestep
                )
                self.update_task_amount_done(self.params.work_rate)
            self.follow(goal.position)

        self.blackboard["TaskExecutingNode"] = Status.RUNNING
        return Status.RUNNING

    def explore(self) -> Status:
        """Look busy by moving to a random imaginary waypoint."""
        if self.exploration_time > self.params.random_exploration_duration:
            self.random_waypoint = get_random_point(self.bounds)
            self.exploration_time = 0

        self.blackboard["random_waypoint"] = self.random_waypoint
        self.exploration_time += self.params.timestep
        self.follow(self.random_waypoint)
        self.blackboard["ExplorationNode"] = Status.RUNNING
        return Status.RUNNING

    async def run_tree(self):
        # Reset action node statuses
        self.blackboard = {
            key: None if key in self.node_callbacks else value
            for key, value in self.blackboard.items()
        }
        return await self.tree.run()

    def follow(self, target: np.ndarray):
        """TODO: rename to update_acceleration? No following here."""
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

    def local_message_receive(self):
        self.agents_nearby = self.get_agents_nearby()
        for other_agent in self.agents_nearby:
            if other_agent.agent_id != self.agent_id:
                self.receive_message(other_agent.message_to_share)
                # other_agent.receive_message(self.message_to_share)

        return self.agents_nearby

    def reset_messages_received(self):
        self.blackboard["messages_received"] = []

    def receive_message(self, message):
        self.blackboard["messages_received"].append(message)

    def set_assigned_task_id(self, task_id):
        self.assigned_task_id = task_id

    def set_planned_tasks(self, task_list):  # This is for visualisation
        self.planned_tasks = task_list

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

    def update_task_amount_done(self, amount):
        self.task_amount_done += amount
