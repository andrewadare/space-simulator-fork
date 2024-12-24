import numpy as np


class Task:
    def __init__(
        self, task_id: int, position: tuple[int, int], radius: float, amount: float
    ):
        self.task_id = task_id
        self.position = np.array(position)
        self.radius = radius
        self.amount = amount
        self.completed = False

    def reduce_amount(self, amount: float):
        self.amount -= amount

        if self.amount <= 0:
            self.completed = True
