import pygame
import math
import random

import modules.behavior_tree as bt
from modules.utils import parse_behavior_tree
from modules.configuration_models import SpaceConfig
from modules.task import Task

agent_track_size = 400  # TODO


def get_decision_making_class(conf: SpaceConfig):
    pass


def get_random_position(x_min, x_max, y_min, y_max):
    return (random.randint(x_min, x_max), random.randint(y_min, y_max))


class Agent:
    def __init__(self, agent_id, position, tasks_info, conf: SpaceConfig):
        self.agent_id = agent_id
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = conf.agents.max_speed
        self.max_accel = conf.agents.max_accel
        self.max_angular_speed = conf.agents.max_angular_speed
        self.work_rate = conf.agents.work_rate
        self.memory_location = []  # TODO: rename to tail; use deque
        self.rotation = 0  # Initial rotation
        self.color = (0, 0, 255)  # Blue color
        self.blackboard = {}
        self.tasks_info: list[Task] = tasks_info  # TODO see README
        self.all_agents: list[Agent] = []  # TODO see README
        self.agents_nearby: list[Agent] = []
        self.communication_radius = conf.agents.communication_radius
        self.situation_awareness_radius = conf.agents.situation_awareness_radius
        self.target_approach_radius = conf.agents.target_approaching_radius
        self.message_to_share = {}
        self.messages_received = []
        self.assigned_task_id = None  # Local decision-making result.
        self.planned_tasks = []  # Local decision-making result.
        self.distance_moved = 0.0
        self.task_amount_done = 0.0
        self.tree = None
        self.task_threshold = conf.tasks.threshold_done_by_arrival
        self.random_exploration_duration = conf.agents.random_exploration_duration
        self.bounds = conf.tasks.locations

        # TODO
        self.timestep = 1.0
        self.task_assigner = get_decision_making_class(conf)

    def sense(self) -> bt.Status:
        """Find waypoints and other agents in this agent's vicinity."""
        self.blackboard["local_tasks_info"] = self.get_tasks_nearby(
            with_completed_task=False
        )
        self.blackboard["local_agents_info"] = self.local_message_receive()
        self.blackboard["LocalSensingNode"] = bt.Status.SUCCESS
        return bt.Status.SUCCESS

    def decide_task(self) -> bt.Status:
        """Decide which waypoint to visit."""
        assigned_task_id = self.task_assigner.decide(self.blackboard, self.timestep)
        status = bt.Status.FAILURE if assigned_task_id is None else bt.Status.SUCCESS
        self.set_assigned_task_id(assigned_task_id)
        self.blackboard["assigned_task_id"] = assigned_task_id
        self.blackboard["DecisionMakingNode"] = status
        return status

    def goto_task(self) -> bt.Status:
        """Go to assigned task position."""
        assigned_task_id = self.blackboard.get("assigned_task_id")

        if assigned_task_id is not None:
            goal: Task = self.tasks_info[assigned_task_id]

            # Check if agent reached the task position
            if (
                self.position.distance_to(goal.position)
                < goal.radius + self.task_threshold
            ):
                if goal.completed:
                    self.blackboard["TaskExecutingNode"] = bt.Status.SUCCESS
                    return bt.Status.SUCCESS
                self.tasks_info[assigned_task_id].reduce_amount(
                    self.work_rate * self.timestep
                )
                self.update_task_amount_done(self.work_rate)
            self.follow(goal.position)

        self.blackboard["TaskExecutingNode"] = bt.Status.RUNNING
        return bt.Status.RUNNING

    def explore(self, _t=float("inf"), _waypoint=(0, 0)) -> bt.Status:
        """Look busy by moving to a random imaginary waypoint.

        Keyword args are used only as static function variables - do not assign.
        """
        # Move towards a random position
        if _t > self.random_exploration_duration:
            _waypoint = get_random_position(
                self.bounds.x_min,
                self.bounds.x_max,
                self.bounds.y_min,
                self.bounds.y_max,
            )
            _t = 0  # Initialisation

        self.blackboard["random_waypoint"] = _waypoint
        _t += self.timestep
        self.follow(_waypoint)
        self.blackboard["ExplorationNode"] = bt.Status.RUNNING
        return bt.Status.RUNNING

    def create_behavior_tree(self):
        self.tree = self._create_behavior_tree()

    def _create_behavior_tree(self) -> bt.Node:
        xml_root = parse_behavior_tree(f"bt_xml/default_bt.xml")
        behavior_tree = self._parse_xml_to_bt(xml_root.find("BehaviorTree"))
        return behavior_tree

    def _parse_xml_to_bt(self, xml_node):
        node_type = xml_node.tag
        children = []

        for child in xml_node:
            children.append(self._parse_xml_to_bt(child))

        if node_type in bt.BehaviorTreeList.CONTROL_NODES:
            control_class = getattr(bt, node_type)
            return control_class(node_type, children=children)
        elif node_type in bt.BehaviorTreeList.ACTION_NODES:
            action_class = getattr(bt, node_type)
            return action_class(node_type, self)
        elif node_type == "BehaviorTree":  # Root
            return children[0]
        else:
            raise ValueError(f"[ERROR] Unknown behavior node type: {node_type}")

    def _reset_bt_action_node_status(self):
        self.blackboard = {
            key: None if key in bt.BehaviorTreeList.ACTION_NODES else value
            for key, value in self.blackboard.items()
        }

    async def run_tree(self):
        self._reset_bt_action_node_status()
        return await self.tree.run(self, self.blackboard)

    def follow(self, target):
        # Calculate desired velocity
        desired = target - self.position
        d = desired.length()

        if d < self.target_approach_radius:

            # Apply arrival behavior
            desired.normalize_ip()

            # Adjust speed based on distance
            desired *= self.max_speed * (d / self.target_approach_radius)
        else:
            desired.normalize_ip()
            desired *= self.max_speed

        steer = desired - self.velocity
        steer = self.limit(steer, self.max_accel)
        self.applyForce(steer)

    def applyForce(self, force):
        self.acceleration += force

    def update(self, sampling_time: float):
        # Update velocity and position
        self.velocity += self.acceleration * sampling_time
        self.velocity = self.limit(self.velocity, self.max_speed)
        self.position += self.velocity * sampling_time
        self.acceleration *= 0  # Reset acceleration

        # Calculate the distance moved in this update and add to distance_moved
        self.distance_moved += self.velocity.length() * sampling_time
        # Memory of positions to draw track
        self.memory_location.append((self.position.x, self.position.y))
        if len(self.memory_location) > agent_track_size:
            self.memory_location.pop(0)

        # Update rotation
        desired_rotation = math.atan2(self.velocity.y, self.velocity.x)
        rotation_diff = desired_rotation - self.rotation
        while rotation_diff > math.pi:
            rotation_diff -= 2 * math.pi
        while rotation_diff < -math.pi:
            rotation_diff += 2 * math.pi

        # Limit angular velocity
        if abs(rotation_diff) > self.max_angular_speed:
            rotation_diff = math.copysign(self.max_angular_speed, rotation_diff)

        self.rotation += rotation_diff * sampling_time

    def reset_movement(self):
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)

    def limit(self, vector, max_value):
        if vector.length_squared() > max_value**2:
            vector.scale_to_length(max_value)
        return vector

    def local_message_receive(self):
        self.agents_nearby = self.get_agents_nearby()
        for other_agent in self.agents_nearby:
            if other_agent.agent_id != self.agent_id:
                self.receive_message(other_agent.message_to_share)
                # other_agent.receive_message(self.message_to_share)

        return self.agents_nearby

    def reset_messages_received(self):
        self.messages_received = []

    def receive_message(self, message):
        self.messages_received.append(message)

    def set_assigned_task_id(self, task_id):
        self.assigned_task_id = task_id

    def set_planned_tasks(self, task_list):  # This is for visualisation
        self.planned_tasks = task_list

    def get_agents_nearby(self, radius=None):
        _communication_radius = self.communication_radius if radius is None else radius
        if _communication_radius > 0:
            communication_radius_squared = _communication_radius**2
            local_agents_info = [
                other_agent
                for other_agent in self.all_agents
                if (self.position - other_agent.position).length_squared()
                <= communication_radius_squared
                and other_agent.agent_id != self.agent_id
            ]
        else:
            local_agents_info = self.all_agents
        return local_agents_info

    def get_tasks_nearby(self, radius=None, with_completed_task=True):
        _situation_awareness_radius = (
            self.situation_awareness_radius if radius is None else radius
        )
        if _situation_awareness_radius > 0:
            situation_awareness_radius_squared = _situation_awareness_radius**2
            if with_completed_task:  # Default
                local_tasks_info = [
                    task
                    for task in self.tasks_info
                    if (self.position - task.position).length_squared()
                    <= situation_awareness_radius_squared
                ]
            else:
                local_tasks_info = [
                    task
                    for task in self.tasks_info
                    if not task.completed
                    and (self.position - task.position).length_squared()
                    <= situation_awareness_radius_squared
                ]
        else:
            if with_completed_task:  # Default
                local_tasks_info = self.tasks_info
            else:
                local_tasks_info = [
                    task for task in self.tasks_info if not task.completed
                ]

        return local_tasks_info

    def update_task_amount_done(self, amount):
        self.task_amount_done += amount
