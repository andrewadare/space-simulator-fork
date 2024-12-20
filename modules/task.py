import pygame  # TODO: remove this dependency


class Task:
    def __init__(self, task_id: int, position, radius: float, amount: float):
        self.task_id = task_id
        self.position = pygame.Vector2(position)
        self.radius = radius
        self.amount = amount
        self.completed = False

    def reduce_amount(self, amount: float):
        self.amount -= amount
        # self.amount -= work_rate * sampling_time
        if self.amount <= 0:
            self.completed = True
