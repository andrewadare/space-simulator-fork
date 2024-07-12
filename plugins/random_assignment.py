import random

class RandomAssignment:
    def __init__(self, agent, tasks_info):
        self.agent = agent
        self.tasks_info = tasks_info

    def decide(self, blackboard):
        '''
        Output: 
            - `task_id`, if task allocation works well
            - `None`, otherwise
        '''        
        if 'assigned_task_id' in blackboard:
            if blackboard['assigned_task_id'] is not None:
                return blackboard['assigned_task_id']
        
        tasks_remaining = [task.task_id for task in self.tasks_info if not task.completed]
        if len(tasks_remaining) > 0:
            return random.choice(tasks_remaining)
        else:
            return None


