from typing import Dict, Any, NamedTuple, Type, Iterable, Tuple

from ..models import (
    GraphTaskModel,
    NodeMulticlassTask,
    GraphRegressionTask,
    GraphBinaryClassificationTask,
    QM9RegressionTask,
)
from ..data import GraphDataset, JsonLGraphPropertyDataset, QM9Dataset, PPIDataset


class TaskInfo(NamedTuple):
    """A named tuple to hold information about a task."""

    name: str
    dataset_class: Type[GraphDataset]
    dataset_default_hypers: Dict[str, Any]
    model_class: Type[GraphTaskModel]
    model_default_hypers: Dict[str, Any]


TASK_NAME_TO_DATASET_AND_MODEL_INFO: Dict[str, TaskInfo] = {}


def register_task(
    task_name, dataset_class, dataset_default_hypers, model_class, model_default_hypers
):
    TASK_NAME_TO_DATASET_AND_MODEL_INFO[task_name.lower()] = TaskInfo(
        name=task_name,
        dataset_class=dataset_class,
        dataset_default_hypers=dataset_default_hypers,
        model_class=model_class,
        model_default_hypers=model_default_hypers,
    )


def clear_known_tasks() -> None:
    TASK_NAME_TO_DATASET_AND_MODEL_INFO.clear()


def get_known_tasks() -> Iterable[str]:
    for task_info in TASK_NAME_TO_DATASET_AND_MODEL_INFO.values():
        yield task_info.name


def task_name_to_dataset_class(name: str) -> Tuple[Type[GraphDataset], Dict[str, Any]]:
    """
    Map task name to a dataset class and default hyperparameters for that class.
    """
    task_info = TASK_NAME_TO_DATASET_AND_MODEL_INFO.get(name.lower())
    if task_info is None:
        raise ValueError("Unknown task type '%s'" % name)
    return task_info.dataset_class, task_info.dataset_default_hypers


def task_name_to_model_class(name: str) -> Tuple[Type[GraphTaskModel], Dict[str, Any]]:
    """
    Map task name to a model class and default hyperparameters for that class.
    """
    task_info = TASK_NAME_TO_DATASET_AND_MODEL_INFO.get(name.lower())
    if task_info is None:
        raise ValueError("Unknown task type '%s'" % name)
    return task_info.model_class, task_info.model_default_hypers


# Register some default tasks:
register_task(
    task_name="PPI",
    dataset_class=PPIDataset,
    dataset_default_hypers={},
    model_class=NodeMulticlassTask,
    model_default_hypers={},
)

register_task(
    task_name="QM9",
    dataset_class=QM9Dataset,
    dataset_default_hypers={},
    model_class=QM9RegressionTask,
    model_default_hypers={},
)

register_task(
    task_name="GraphRegression",
    dataset_class=JsonLGraphPropertyDataset,
    dataset_default_hypers={"threshold_for_classification": None},
    model_class=GraphRegressionTask,
    model_default_hypers={},
)

register_task(
    task_name="GraphBinaryClassification",
    dataset_class=JsonLGraphPropertyDataset,
    dataset_default_hypers={"threshold_for_classification": 23.0},
    model_class=GraphBinaryClassificationTask,
    model_default_hypers={},
)
