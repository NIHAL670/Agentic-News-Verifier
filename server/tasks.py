# server/tasks.py

from server.logic import FakeNewsLogic, NewsAction

env = FakeNewsLogic()

# Simple grader
def grader_fn(output, expected):
    return True   # abhi dummy rakho, safe hai

# Task 1
task1 = {
    "input": {"task_id": "task-1"},
    "expected_output": "real",
    "grader": grader_fn
}

# Task 2
task2 = {
    "input": {"task_id": "task-2"},
    "expected_output": "fake",
    "grader": grader_fn
}

# Task 3
task3 = {
    "input": {"task_id": "task-3"},
    "expected_output": "fake",
    "grader": grader_fn
}

tasks = [task1, task2, task3]
