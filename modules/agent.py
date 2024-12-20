import pygame
import math

import modules.behavior_tree as bt
from modules.utils import parse_behavior_tree
from modules.configuration_models import AgentConfig
from modules.task import Task

agent_track_size = 400


class Agent:
    def __init__(self, agent_id, position, tasks_info, conf: AgentConfig):
        self.agent_id = agent_id
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)
        self.max_speed = conf.max_speed
        self.max_accel = conf.max_accel
        self.max_angular_speed = conf.max_angular_speed
        self.work_rate = conf.work_rate
        self.memory_location = []  # To draw track
        self.rotation = 0  # Initial rotation
        self.color = (0, 0, 255)  # Blue color
        self.blackboard = {}
        self.tasks_info: list[Task] = tasks_info  # TODO see README
        self.all_agents: list[Agent] = []  # TODO see README
        self.agents_nearby: list[Agent] = []
        self.communication_radius = conf.communication_radius
        self.situation_awareness_radius = conf.situation_awareness_radius
        self.target_approach_radius = conf.target_approaching_radius
        self.message_to_share = {}
        self.messages_received = []
        self.assigned_task_id = None  # Local decision-making result.
        self.planned_tasks = []  # Local decision-making result.
        self.distance_moved = 0.0
        self.task_amount_done = 0.0
        self.tree = None

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
