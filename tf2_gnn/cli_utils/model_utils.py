import json
import os
import pickle
from typing import Dict, Any, Optional, Set, Type, Callable

import h5py
import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath
from tensorflow.python.keras import backend as K

from ..data import DataFold, GraphDataset
from ..models import GraphTaskModel
from .dataset_utils import get_dataset, get_model_file_path
from .task_utils import task_name_to_model_class
from .param_helpers import override_model_params_with_hyperdrive_params


def _get_name_to_variable_map(model: GraphTaskModel) -> Dict[str, tf.Variable]:
    var_name_to_variable: Dict[str, tf.Variable] = {}
    var_names_unique = True
    for var in model.variables:
        if var.name in var_name_to_variable:
            print(
                f"E: More than one variable with name {var.name} used in model. Please use appropriate name_scopes!"
            )
            var_names_unique = False
        else:
            var_name_to_variable[var.name] = var
    if not var_names_unique:
        raise ValueError(
            "Model variables have duplicate names, making weight restoring impossible."
        )
    return var_name_to_variable


def save_model(
    save_file: str,
    model: GraphTaskModel,
    dataset: GraphDataset,
    extra_data_to_store: Dict[str, Any] = {},
    store_weights_in_pkl: bool = False,
) -> None:
    data_to_store = {
        "model_class": model.__class__,
        "model_params": model._params,
        "dataset_class": dataset.__class__,
        "dataset_params": dataset._params,
        "dataset_metadata": dataset._metadata,
        "num_edge_types": dataset.num_edge_types,
        "node_feature_shape": dataset.node_feature_shape,
    }
    if store_weights_in_pkl:
        var_name_to_variable = _get_name_to_variable_map(model)
        var_name_to_weights = {
            name: var.value().numpy() for name, var in var_name_to_variable.items()
        }
        data_to_store["model_weights"] = var_name_to_weights

    data_to_store.update(extra_data_to_store)

    pkl_file = get_model_file_path(save_file, "pkl")
    with open(pkl_file, "wb") as out_file:
        pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)

    if store_weights_in_pkl:
        print(f"   (Stored model metadata and weights to {pkl_file}).")
    else:
        hdf5_file = get_model_file_path(save_file, "hdf5")
        model.save_weights(hdf5_file, save_format="h5")
        print(f"   (Stored model metadata to {pkl_file} and weights to {hdf5_file})")


def _read_weights_from_hdf5(save_file):
    hdf5_save_file = get_model_file_path(save_file, "hdf5")
    var_name_to_weights = {}

    def hdf5_item_visitor(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        if name in var_name_to_weights:
            raise ValueError(
                f"More than one variable with name {name} used in hdf5 file. Please use appropriate name_scopes!"
            )
        else:
            var_name_to_weights[name] = np.array(item)

    with h5py.File(hdf5_save_file, mode="r") as data_hdf5:
        # For some reason, the first layer of attributes is auto-generated names instead of actual names:
        for model_sublayer in data_hdf5.values():
            model_sublayer.visititems(hdf5_item_visitor)

    return var_name_to_weights


# This map helps to load weights from old version of the library in cases where we have changed
# the naming of variables inbetween versions:
BACKWARD_COMPAT_WEIGHT_NAME_MAP = {
    "/Global_Exchange/graph_global_mean_exchange/": "/Global_Exchange/GraphGlobalMeanExchange/",
    "/Global_Exchange/graph_global_gru_exchange/": "/Global_Exchange/GraphGlobalGRUExchange/",
    "/Global_Exchange/graph_global_mlp_exchange/": "/Global_Exchange/GraphGlobalMLPExchange/",
}


def backward_compat_weight_renaming_fn(weight_name: str) -> str:
    for old_name, new_name in BACKWARD_COMPAT_WEIGHT_NAME_MAP.items():
        weight_name = weight_name.replace(old_name, new_name)
    return weight_name


def load_weights_verbosely(
    save_file: str,
    model: GraphTaskModel,
    warn_about_initialisations: bool = True,
    warn_about_ignored: bool = True,
    weight_name_to_var_name: Optional[Callable[[str], str]] = backward_compat_weight_renaming_fn,
):
    var_name_to_variable = _get_name_to_variable_map(model)

    with open(get_model_file_path(save_file, "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)
    var_name_to_weights = data_to_load.get("model_weights")
    if var_name_to_weights is None:
        var_name_to_weights = _read_weights_from_hdf5(save_file)

    if weight_name_to_var_name is not None:
        remapped_var_name_to_weights = {}
        for weight_name, weight in var_name_to_weights.items():
            remapped_var_name_to_weights[weight_name_to_var_name(weight_name)] = weight
        var_name_to_weights = remapped_var_name_to_weights

    tfvar_weight_tuples = []
    used_var_names = set()
    for var_name, tfvar in var_name_to_variable.items():
        saved_weight = var_name_to_weights.get(var_name)
        if saved_weight is None:
            if warn_about_initialisations:
                print(f"I: Weights for {var_name} freshly initialised.")
        else:
            used_var_names.add(var_name)
            tfvar_weight_tuples.append((tfvar, saved_weight))

    if warn_about_ignored:
        for var_name in var_name_to_weights.keys():
            if var_name not in used_var_names:
                print(f"I: Model does not use saved weights for {var_name}.")

    K.batch_set_value(tfvar_weight_tuples)


def load_dataset_for_prediction(trained_model_file: str):
    with open(get_model_file_path(trained_model_file, "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)
    dataset_class: Type[GraphDataset] = data_to_load["dataset_class"]

    return dataset_class(
        params=data_to_load.get("dataset_params", {}),
        metadata=data_to_load.get("dataset_metadata", {}),
    )


def load_model_for_prediction(
    trained_model_file: str, dataset: GraphDataset, disable_tf_function_build: bool = False
):
    with open(get_model_file_path(trained_model_file, "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)
    model_class: Type[GraphTaskModel] = data_to_load["model_class"]

    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    model = model_class(
        params=data_to_load.get("model_params", {}),
        dataset=dataset,
        disable_tf_function_build=disable_tf_function_build,
    )

    data_description = dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    print(f"Restoring model weights from {trained_model_file}.")
    load_weights_verbosely(trained_model_file, model)

    return model


def get_model(
    msg_passing_implementation: str,
    task_name: str,
    model_cls: Type[GraphTaskModel],
    dataset: GraphDataset,
    dataset_model_optimised_default_hyperparameters: Dict[str, Any],
    loaded_model_hyperparameters: Dict[str, Any],
    cli_model_hyperparameter_overrides: Dict[str, Any],
    hyperdrive_hyperparameter_overrides: Dict[str, str],
    disable_tf_function_build: bool = False,
) -> GraphTaskModel:
    if not model_cls:
        model_cls, model_default_hyperparameter_overrides = task_name_to_model_class(task_name)
        model_params = model_cls.get_default_hyperparameters(msg_passing_implementation)
        print(f" Model default parameters: {model_params}")
        model_params.update(model_default_hyperparameter_overrides)
        if len(model_default_hyperparameter_overrides):
            print(
                f"  Model parameters overridden by task defaults: {model_default_hyperparameter_overrides}"
            )
        model_params.update(dataset_model_optimised_default_hyperparameters)
        if len(dataset_model_optimised_default_hyperparameters):
            print(
                f"  Model parameters overridden by task/model defaults: {dataset_model_optimised_default_hyperparameters}"
            )
    else:
        model_params = loaded_model_hyperparameters
    model_params.update(cli_model_hyperparameter_overrides)
    if len(cli_model_hyperparameter_overrides):
        print(f"  Model parameters overridden from CLI: {cli_model_hyperparameter_overrides}")
    if len(hyperdrive_hyperparameter_overrides) > 0:
        override_model_params_with_hyperdrive_params(
            model_params, hyperdrive_hyperparameter_overrides
        )
        print(
            f"  Model parameters overridden for Hyperdrive: {hyperdrive_hyperparameter_overrides}"
        )
    return model_cls(
        model_params, dataset=dataset, disable_tf_function_build=disable_tf_function_build
    )


# TODO: A better solution to 'loading weights only without model and class' is required.
# In particular, need to ensure that the weights and the proposed model to be trained match up in their
# base components (i.e. if loading GNN weights, dimensions need to match for the GNN in finetuning)
def get_model_and_dataset(
    task_name: Optional[str],
    msg_passing_implementation: Optional[str],
    data_path: RichPath,
    trained_model_file: Optional[str],
    cli_data_hyperparameter_overrides: Optional[str],
    cli_model_hyperparameter_overrides: Optional[str],
    hyperdrive_hyperparameter_overrides: Dict[str, str] = {},
    folds_to_load: Optional[Set[DataFold]] = None,
    load_weights_only: bool = False,
    disable_tf_function_build: bool = False,
):
    if trained_model_file and not os.path.exists(trained_model_file):
        print(f"W: Asked to load from {trained_model_file}, which does not exist. Ignoring.")
        trained_model_file = None

    # case of a trained model file being passed, where the entire model should be loaded,
    # a new class and dataset type are not required
    if trained_model_file and not load_weights_only:
        with open(get_model_file_path(trained_model_file, "pkl"), "rb") as in_file:
            data_to_load = pickle.load(in_file)
        model_class = data_to_load["model_class"]
        dataset_class = data_to_load["dataset_class"]
        default_task_model_hypers = {}
    # case 1: trained_model_file and loading weights only -- create new dataset and class of the type specified by the
    # task to be trained, but use weights from another corresponding model
    # case 2: no model to be loaded; make fresh dataset and model classes
    elif (trained_model_file and load_weights_only) or not trained_model_file:
        data_to_load = {}
        model_class, dataset_class = None, None

        # Load potential task-specific defaults:
        default_task_model_hypers = {}
        task_model_default_hypers_file = os.path.join(
            os.path.dirname(__file__),
            "default_hypers",
            "%s_%s.json" % (task_name, msg_passing_implementation),
        )
        print(
            f"Trying to load task/model-specific default parameters from {task_model_default_hypers_file} ... ",
            end="",
        )
        if os.path.exists(task_model_default_hypers_file):
            print("File found.")
            with open(task_model_default_hypers_file, "rt") as f:
                default_task_model_hypers = json.load(f)
        else:
            print("File not found, using global defaults.")

        if not trained_model_file and load_weights_only:
            raise ValueError(
                "Cannot load only weights when model file from which to load is not specified."
            )

    dataset = get_dataset(
        task_name,
        dataset_class,
        default_task_model_hypers.get("task_params", {}),
        data_to_load.get("dataset_params", {}),
        json.loads(cli_data_hyperparameter_overrides or "{}"),
        data_to_load.get("dataset_metadata", {}),
    )

    # Actually load data:
    print(f"Loading data from {data_path}.")
    dataset.load_data(data_path, folds_to_load)

    model = get_model(
        msg_passing_implementation,
        task_name,
        model_class,
        dataset,
        dataset_model_optimised_default_hyperparameters=default_task_model_hypers.get(
            "model_params", {}
        ),
        loaded_model_hyperparameters=data_to_load.get("model_params", {}),
        cli_model_hyperparameter_overrides=json.loads(cli_model_hyperparameter_overrides or "{}"),
        hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides or {},
        disable_tf_function_build=disable_tf_function_build,
    )

    data_description = dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    # If needed, load weights for model:
    if trained_model_file:
        print(f"Restoring model weights from {trained_model_file}.")
        load_weights_verbosely(trained_model_file, model)

    return dataset, model
