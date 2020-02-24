import pickle
from typing import Dict, Any, Optional, Type

from ..data import GraphDataset
from .task_utils import task_name_to_dataset_class


# This should rightfully live in model_utils.py, but that would lead to a circular import...
def get_model_file_path(model_path: str, target_suffix: str):
    assert target_suffix in ("hdf5", "pkl")
    if model_path.endswith(".hdf5"):
        return model_path[:-4] + target_suffix
    elif model_path.endswith(".pkl"):
        return model_path[:-3] + target_suffix
    else:
        raise ValueError(
            f"Model path has to end in hdf5/pkl, which is not the case for {model_path}!"
        )


def load_dataset_for_prediction(trained_model_file: str):
    with open(get_model_file_path(trained_model_file, "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)
    dataset_class: Type[GraphDataset] = data_to_load["dataset_class"]

    return dataset_class(
        params=data_to_load.get("dataset_params", {}),
        metadata=data_to_load.get("dataset_metadata", {}),
    )


def get_dataset(
    task_name: Optional[str],
    dataset_cls: Optional[Type[GraphDataset]],
    dataset_model_optimised_default_hyperparameters: Dict[str, Any],
    loaded_data_hyperparameters: Dict[str, Any],
    cli_data_hyperparameter_overrides: Dict[str, Any],
    loaded_metadata: Dict[str, Any],
) -> GraphDataset:
    if not dataset_cls:
        (
            dataset_cls,
            dataset_default_hyperparameter_overrides,
        ) = task_name_to_dataset_class(task_name)
        dataset_params = dataset_cls.get_default_hyperparameters()
        print(f" Dataset default parameters: {dataset_params}")
        dataset_params.update(dataset_default_hyperparameter_overrides)
        if len(dataset_default_hyperparameter_overrides):
            print(
                f"  Dataset parameters overridden by task defaults: {dataset_default_hyperparameter_overrides}"
            )
        dataset_params.update(dataset_model_optimised_default_hyperparameters)
        if len(dataset_default_hyperparameter_overrides):
            print(
                f"  Dataset parameters overridden by task/model defaults: {dataset_model_optimised_default_hyperparameters}"
            )
    else:
        dataset_params = loaded_data_hyperparameters
    dataset_params.update(cli_data_hyperparameter_overrides)
    if len(cli_data_hyperparameter_overrides):
        print(
            f"  Dataset parameters overridden from CLI: {cli_data_hyperparameter_overrides}"
        )
    if len(loaded_metadata):
        print("  WARNING: Dataset metadata loaded from disk, not calculated from data.")
    return dataset_cls(dataset_params, loaded_metadata)
