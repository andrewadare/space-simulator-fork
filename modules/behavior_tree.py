from enum import Enum
from typing import Protocol


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
        result: Status = self.action()
        # blackboard[self.name] = result
        return result
