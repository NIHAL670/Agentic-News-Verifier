# server/tasks.py

def grader1(output, expected):
    return str(output).strip().lower() == str(expected).strip().lower()

def grader2(output, expected):
    return str(output).strip().lower() == str(expected).strip().lower()

def grader3(output, expected):
    return str(output).strip().lower() == str(expected).strip().lower()

task1 = {
    "input": "task-1",
    "expected_output": "false",
    "name": "Easy: Historical Fact",
    "grader": grader1
}

task2 = {
    "input": "task-2",
    "expected_output": "true",
    "name": "Medium: Current Events",
    "grader": grader2
}

task3 = {
    "input": "task-3",
    "expected_output": "false",
    "name": "Hard: Contextual Misinformation",
    "grader": grader3
}

tasks = [task1, task2, task3]
