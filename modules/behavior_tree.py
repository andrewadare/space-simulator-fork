from enum import Enum
from typing import Protocol
from pathlib import Path
import xml.etree.ElementTree as ET


# Status enumeration for behavior tree nodes
class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class ReturnsStatus(Protocol):
    def __call__(self) -> Status: ...


# Base class for all behavior tree nodes
class Node:
    def __init__(self, name: str):
        self.name = name

    async def run(self) -> Status: ...


# Sequence node: Runs child nodes in sequence until one fails
class SequenceNode(Node):
    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = children

    async def run(self) -> Status:
        for child in self.children:
            status = await child.run()
            if status == Status.RUNNING:
                continue
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


# Fallback node: Runs child nodes in sequence until one succeeds
class FallbackNode(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    async def run(self) -> Status:
        for child in self.children:
            status = await child.run()
            if status == Status.RUNNING:
                continue
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


# Synchronous action node
class SyncActionNode(Node):
    def __init__(self, name, action: ReturnsStatus):
        super().__init__(name)
        self.action = action

    async def run(self) -> Status:
        return self.action()


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
