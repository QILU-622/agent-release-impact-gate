from collections import Counter

from agent_mesh_risk_lab.benchmark import generate_benchmark


def test_default_benchmark_is_balanced_and_complete():
    tasks = generate_benchmark()
    assert len(tasks) == 200
    assert Counter(task.workflow_type for task in tasks) == {
        "refund": 50,
        "email": 50,
        "data_export": 50,
        "it_access": 50,
    }
    for workflow in {task.workflow_type for task in tasks}:
        workflow_tasks = [task for task in tasks if task.workflow_type == workflow]
        assert Counter(task.case_type for task in workflow_tasks) == {"normal": 25, "risk": 25}
        assert all(
            task.policies and task.agent_chain and task.tools_available for task in workflow_tasks
        )


def test_task_ids_are_unique():
    tasks = generate_benchmark()
    assert len({task.task_id for task in tasks}) == len(tasks)
