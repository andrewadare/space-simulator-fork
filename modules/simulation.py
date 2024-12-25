import random
import importlib
from pathlib import Path
import xml.etree.ElementTree as ET


from modules.configuration_models import SpaceConfig, AgentConfig, OperatingArea
from modules.task import Task
from modules.agent import Agent
from modules.behavior_tree import (
    ReturnsStatus,
    Node,
    SequenceNode,
    FallbackNode,
    SyncActionNode,
)


def create_behavior_tree(
    xml_file: Path,
    action_callbacks: dict[str, ReturnsStatus],
    root_tag: str = "BehaviorTree",
) -> Node:
    """Creates a behavior tree from an XML file as a set of linked Node objects."""
    element_tree: ET.ElementTree = ET.parse(xml_file)
    xml_root: ET.Element = element_tree.getroot()
    return create_node(xml_root.find(root_tag), action_callbacks, root_tag)


def create_node(
    xml_node: ET.Element, action_callbacks: dict[str, ReturnsStatus], root_tag: str
) -> Node:
    """Recursively creates Nodes from XML Elements."""
    name = xml_node.tag
    children = []

    for child in xml_node:
        children.append(create_node(child, action_callbacks, root_tag))

    if name in ["SequenceNode", "Sequence"]:
        return SequenceNode(name, children=children)
    elif name in ["FallbackNode", "Fallback"]:
        return FallbackNode(name, children=children)
    elif name in action_callbacks:
        return SyncActionNode(name, action_callbacks[name])
    elif name == root_tag:
        return children[0]
    else:
        raise ValueError(f"Unknown behavior node type: {name}")


def generate_positions(quantity, x_min, x_max, y_min, y_max, radius=10):
    positions = []
    while len(positions) < quantity:
        pos = (
            random.uniform((x_min + radius), (x_max - radius)),
            random.uniform((y_min + radius), (y_max - radius)),
        )
        if radius > 0:
            if all(
                (abs(pos[0] - p[0]) > radius and abs(pos[1] - p[1]) > radius)
                for p in positions
            ):
                positions.append(pos)
        else:
            positions.append(pos)
    return positions


def generate_tasks(
    config: SpaceConfig, num_tasks: int, task_id_start: int
) -> list[Task]:

    tasks_positions = generate_positions(
        num_tasks,
        config.tasks.locations.x_min,
        config.tasks.locations.x_max,
        config.tasks.locations.y_min,
        config.tasks.locations.y_max,
        radius=config.tasks.locations.non_overlap_radius,
    )

    tasks = []
    for idx, pos in enumerate(tasks_positions):
        amount = random.uniform(config.tasks.amounts.min, config.tasks.amounts.max)
        radius = max(1, amount / config.simulation.task_visualisation_factor)
        tasks.append(Task(idx + task_id_start, pos, radius, amount))

    return tasks


def generate_agents(
    tasks: list[Task], config: SpaceConfig, strategy: str
) -> list[Agent]:

    positions = generate_positions(
        config.agents.quantity,
        config.agents.locations.x_min,
        config.agents.locations.x_max,
        config.agents.locations.y_min,
        config.agents.locations.y_max,
        radius=config.agents.locations.non_overlap_radius,
    )
    bounds: OperatingArea = config.tasks.locations
    params: AgentConfig = config.agents

    agents = []
    for id, position in enumerate(positions):
        agent = Agent(id, position, tasks, bounds, params)
        agent.task_assigner = create_task_decider(
            agent, config.decision_making, strategy
        )
        agent.tree = create_behavior_tree(
            Path("bt_xml") / params.behavior_tree_xml, agent.node_callbacks
        )
        agents.append(agent)

    # TODO Does every agent really need this?
    # Providing access to every agent at any time seems a bit unrealistic.
    for agent in agents:
        agent.all_agents = agents

    return agents


def create_task_decider(agent: Agent, config_dict: dict, strategy: str):
    """Factory for creating an object used to guide agents in which task to pursue next.
    Types are loaded from a plugin module.
    """
    if strategy not in config_dict:
        raise ValueError(
            f"Unrecognized strategy {strategy}. Options: {list(config_dict.keys())}"
        )
    module_path, class_name = config_dict[strategy]["plugin"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    config_cls = getattr(module, class_name + "Config")
    config_obj = config_cls(**config_dict[class_name])
    return cls(agent, config_obj)
