import json
import os
import pickle
import random
import sys
import time
from typing import List, Dict, Any, Optional, Callable, Set, Type

import h5py
import numpy as np
import tensorflow as tf
from dpu_utils.utils import run_and_debug, RichPath
from tensorflow.python.keras import backend as K

from tf2_gnn import DataFold, GraphDataset, GraphTaskModel, get_known_message_passing_classes
from .task_utils import get_known_tasks, task_name_to_dataset_class, task_name_to_model_class


def make_run_id(model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))


def save_model(save_file, model: GraphTaskModel, dataset: GraphDataset) -> None:
    data_to_store = {
        "model_class": model.__class__,
        "model_params": model._params,
        "dataset_class": dataset.__class__,
        "dataset_params": dataset._params,
        "dataset_metadata": dataset._metadata,
    }
    with open(save_file, "wb") as out_file:
        pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)
    hdf5_file = save_file[:-3] + "hdf5"
    model.save_weights(hdf5_file, save_format="h5")
    print(f"   (Stored model metadata to {save_file} and weights to {hdf5_file})")


def load_weights_verbosely(model_weights_file: str, model: GraphTaskModel):
    var_name_to_variable = {}
    var_names_unique = True
    for var in model.variables:
        if var.name in var_name_to_variable:
            print(f"E: More than one variable with name {var.name} used in model. Please use appropriate name_scopes!")
            var_names_unique = False
        else:
            var_name_to_variable[var.name] = var
    if not var_names_unique:
        raise ValueError("Model variables have duplicate names, making weight restoring impossible.")

    var_name_to_weights = {}
    def hdf5_item_visitor(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        if name in var_name_to_weights:
            print(f"E: More than one variable with name {name} used in hdf5 file. Please use appropriate name_scopes!")
            var_names_unique = False
        else:
            var_name_to_weights[name] = np.array(item)

    with h5py.File(model_weights_file, mode='r') as data_hdf5:
        # For some reason, the first layer of attributes is auto-generated names instead of actual names:
        for model_sublayer in data_hdf5.values():
            model_sublayer.visititems(hdf5_item_visitor)
    if not var_names_unique:
        raise ValueError("Stored weights have duplicate names, making weight restoring impossible.")

    tfvar_weight_tuples = []
    for var_name, tfvar in var_name_to_variable.items():
        saved_weight = var_name_to_weights.get(var_name)
        if saved_weight is None:
            print(f"I: Weights for {var_name} freshly initialised.")
        else:
            tfvar_weight_tuples.append((tfvar, saved_weight))

    for var_name in var_name_to_weights.keys():
        if var_name not in var_name_to_variable:
            print(f"I: Model does not use saved weights for {var_name}.")

    K.batch_set_value(tfvar_weight_tuples)


def get_dataset(
    task_name: Optional[str],
    dataset_cls: Optional[Type[GraphDataset]],
    dataset_model_optimised_default_hyperparameters: Dict[str, Any],
    loaded_data_hyperparameters: Dict[str, Any],
    cli_data_hyperparameter_overrides: Dict[str, Any],
    loaded_metadata: Dict[str, Any],
) -> GraphDataset:
    if not dataset_cls:
        dataset_cls, dataset_default_hyperparameter_overrides = task_name_to_dataset_class(task_name)
        dataset_params = dataset_cls.get_default_hyperparameters()
        print(f" Dataset default parameters: {dataset_params}")
        dataset_params.update(dataset_default_hyperparameter_overrides)
        if len(dataset_default_hyperparameter_overrides):
            print(f"  Dataset parameters overridden by task defaults: {dataset_default_hyperparameter_overrides}")
        dataset_params.update(dataset_model_optimised_default_hyperparameters)
        if len(dataset_default_hyperparameter_overrides):
            print(f"  Dataset parameters overridden by task/model defaults: {dataset_model_optimised_default_hyperparameters}")
    else:
        dataset_params = loaded_data_hyperparameters
    dataset_params.update(cli_data_hyperparameter_overrides)
    if len(cli_data_hyperparameter_overrides):
        print(f"  Dataset parameters overridden from CLI: {cli_data_hyperparameter_overrides}")
    if len(loaded_metadata):
        print("  WARNING: Dataset metadata loaded from disk, not calculated from data.")
    return dataset_cls(dataset_params, loaded_metadata)


def get_model(
    msg_passing_implementation: str,
    task_name: str,
    model_cls: Type[GraphTaskModel],
    dataset: GraphDataset,
    dataset_model_optimised_default_hyperparameters: Dict[str, Any],
    loaded_model_hyperparameters: Dict[str, Any],
    cli_model_hyperparameter_overrides: Dict[str, Any],
    hyperdrive_hyperparameter_overrides: Dict[str, str],
) -> GraphTaskModel:
    if not model_cls:
        model_cls, model_default_hyperparameter_overrides = task_name_to_model_class(task_name)
        model_params = model_cls.get_default_hyperparameters(msg_passing_implementation)
        print(f" Model default parameters: {model_params}")
        model_params.update(model_default_hyperparameter_overrides)
        if len(model_default_hyperparameter_overrides):
            print(f"  Model parameters overridden by task defaults: {model_default_hyperparameter_overrides}")
        model_params.update(dataset_model_optimised_default_hyperparameters)
        if len(dataset_model_optimised_default_hyperparameters):
            print(f"  Model parameters overridden by task/model defaults: {dataset_model_optimised_default_hyperparameters}")
    else:
        model_params = loaded_model_hyperparameters
    model_params.update(cli_model_hyperparameter_overrides)
    if len(cli_model_hyperparameter_overrides):
        print(f"  Model parameters overridden from CLI: {cli_model_hyperparameter_overrides}")
    if len(hyperdrive_hyperparameter_overrides) > 0:
        # Only require azure_ml if needed:
        from ..azure_ml.utils import override_model_params_with_hyperdrive_params
        override_model_params_with_hyperdrive_params(model_params, hyperdrive_hyperparameter_overrides)
        print(f"  Model parameters overridden for Hyperdrive: {hyperdrive_hyperparameter_overrides}")
    return model_cls(model_params, dataset=dataset)


def load_dataset_for_prediction(trained_model_file: str):
    with open(trained_model_file, "rb") as in_file:
        data_to_load = pickle.load(in_file)
    dataset_class : Type[GraphDataset] = data_to_load["dataset_class"]

    return dataset_class(
        params=data_to_load.get("dataset_params", {}),
        metadata=data_to_load.get("dataset_metadata", {}),
    )


def load_model_for_prediction(trained_model_file: str, dataset: GraphDataset):
    with open(trained_model_file, "rb") as in_file:
        data_to_load = pickle.load(in_file)
    model_class : Type[GraphTaskModel] = data_to_load["model_class"]

    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    model = model_class(
        params=data_to_load.get("model_params", {}),
        dataset=dataset,
    )

    data_description = dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    trained_model_weights_file = trained_model_file[:-3] + "hdf5"
    print(f"Restoring model weights from {trained_model_weights_file}.")
    load_weights_verbosely(trained_model_weights_file, model)

    return model


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
):
    # case of a trained model file being passed, where the entire model should be loaded, 
    # a new class and dataset type are not required
    if trained_model_file and not load_weights_only:
        with open(trained_model_file, "rb") as in_file:
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
            os.path.dirname(__file__), "default_hypers", "%s_%s.json" % (task_name, msg_passing_implementation)
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
            raise ValueError("Cannot load only weights when model file from which to load is not specified.")

    dataset = get_dataset(
        task_name,
        dataset_class,
        default_task_model_hypers.get("task_params", {}),
        data_to_load.get("dataset_params", {}),
        json.loads(cli_data_hyperparameter_overrides or "{}"),
        data_to_load.get("dataset_metadata", {})
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
    )

    data_description = dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    # If needed, load weights for model:
    if trained_model_file:
        trained_model_weights_file = trained_model_file[:-3] + "hdf5"
        print(f"Restoring model weights from {trained_model_weights_file}.")
        load_weights_verbosely(trained_model_weights_file, model)

    return dataset, model


def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    print(msg)


def train(
    model: GraphTaskModel,
    dataset: GraphDataset,
    log_fun: Callable[[str], None],
    run_id: str,
    max_epochs: int,
    patience: int,
    save_dir: str,
    quiet: bool = False,
    aml_run=None,
):
    train_data = dataset.get_tensorflow_dataset(DataFold.TRAIN).prefetch(3)
    valid_data = dataset.get_tensorflow_dataset(DataFold.VALIDATION).prefetch(3)

    save_file = os.path.join(save_dir, f"{run_id}_best.pkl")

    _, _, initial_valid_results = model.run_one_epoch(valid_data, training=False, quiet=quiet)
    best_valid_metric, best_val_str = model.compute_epoch_metrics(initial_valid_results)
    log_fun(f"Initial valid metric: {best_val_str}.")
    save_model(save_file, model, dataset)
    best_valid_epoch = 0
    train_time_start = time.time()
    for epoch in range(1, max_epochs + 1):
        log_fun(f"== Epoch {epoch}")
        train_loss, train_speed, train_results = model.run_one_epoch(
            train_data, training=True, quiet=quiet
        )
        train_metric, train_metric_string = model.compute_epoch_metrics(train_results)
        log_fun(
            f" Train:  {train_loss:.4f} loss | {train_metric_string} | {train_speed:.2f} graphs/s",
        )
        valid_loss, valid_speed, valid_results = model.run_one_epoch(
            valid_data, training=False, quiet=quiet
        )
        valid_metric, valid_metric_string = model.compute_epoch_metrics(valid_results)
        log_fun(
            f" Valid:  {valid_loss:.4f} loss | {valid_metric_string} | {valid_speed:.2f} graphs/s",
        )

        if aml_run is not None:
            aml_run.log("task_train_metric", float(train_metric))
            aml_run.log("train_speed", float(train_speed))
            aml_run.log("task_valid_metric", float(valid_metric))
            aml_run.log("valid_speed", float(valid_speed))

        # Save if good enough.
        if valid_metric < best_valid_metric:
            log_fun(
                f"  (Best epoch so far, target metric decreased to {valid_metric:.5f} from {best_valid_metric:.5f}.)",
            )
            save_model(save_file, model, dataset)
            best_valid_metric = valid_metric
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            log_fun(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation metric.",
            )
            log_fun(f"Training took {total_time}s. Best validation metric: {best_valid_metric}",)
            break
    return save_file

def run_train_from_args(args, hyperdrive_hyperparameter_overrides: Dict[str, str] = {}) -> None:
    # Get the housekeeping going and start logging:
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = make_run_id(args.model, args.task)
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    def log(msg):
        log_line(log_file, msg)

    log(f"Setting random seed {args.random_seed}.")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    data_path = RichPath.create(args.data_path, args.azure_info)
    try:
        dataset, model = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=data_path,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides,
            folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
            load_weights_only=args.load_weights_only,
        )
    except ValueError as err:
        print(err.args)

    log(f"Dataset parameters: {json.dumps(dict(dataset._params))}")
    log(f"Model parameters: {json.dumps(dict(model._params))}")

    if args.azureml_logging:
        from azureml.core.run import Run

        aml_run = Run.get_context()
    else:
        aml_run = None

    trained_model_path = train(
        model,
        dataset,
        log_fun=log,
        run_id=run_id,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=args.save_dir,
        quiet=args.quiet,
        aml_run=aml_run,
    )

    if args.run_test:
        data_path = RichPath.create(args.data_path, args.azure_info)
        log("== Running on test dataset")
        log(f"Loading data from {data_path}.")
        dataset.load_data(data_path, {DataFold.TEST})
        log(f"Restoring best model state from {trained_model_path}.")
        load_weights_verbosely(trained_model_path[:-3] + "hdf5", model)
        test_data = dataset.get_tensorflow_dataset(DataFold.TEST)
        _, _, test_results = model.run_one_epoch(test_data, training=False, quiet=args.quiet)
        test_metric, test_metric_string = model.compute_epoch_metrics(test_results)
        log(test_metric_string)
        if aml_run is not None:
            aml_run.log("task_test_metric", float(test_metric))


def get_train_cli_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model.")
    # We use a somewhat horrible trick to support both
    #  train.py --model MODEL --task TASK --data_path DATA_PATH
    # as well as
    #  train.py model task data_path
    # The former is useful because of limitations in AzureML; the latter is nicer to type.
    if "--model" in sys.argv:
        model_param_name, task_param_name, data_path_param_name = "--model", "--task", "--data_path"
    else:
        model_param_name, task_param_name, data_path_param_name = "model", "task", "data_path"

    parser.add_argument(
        model_param_name,
        type=str,
        choices=sorted(get_known_message_passing_classes()),
        help="GNN model type to train.",
    )
    parser.add_argument(
        task_param_name,
        type=str,
        choices=sorted(get_known_tasks()),
        help="Task to train model for.",
    )
    parser.add_argument(data_path_param_name, type=str, help="Directory containing the task data.")
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="trained_model",
        help="Path in which to store the trained model and log.",
    )
    parser.add_argument(
        "--model-params-override",
        dest="model_param_override",
        type=str,
        help="JSON dictionary overriding model hyperparameter values.",
    )
    parser.add_argument(
        "--data-params-override",
        dest="data_param_override",
        type=str,
        help="JSON dictionary overriding data hyperparameter values.",
    )
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=10000,
        help="Maximal number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=25,
        help="Maximal number of epochs to continue training without improvement.",
    )
    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use.",
    )
    parser.add_argument(
        "--run-name", dest="run_name", type=str, help="A human-readable name for this run.",
    )
    parser.add_argument(
        "--azure-info",
        dest="azure_info",
        type=str,
        default="azure_auth.json",
        help="Azure authentication information file (JSON).",
    )
    parser.add_argument(
        "--load-saved-model",
        dest="load_saved_model",
        help="Optional location to load initial model weights from. Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--load-weights-only",
        dest="load_weights_only",
        action="store_true",
        help="Optional to only load the weights of the model rather than class and dataset for further training (used in fine-tuning on pretrained network). Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--quiet", dest="quiet", action="store_true", help="Generate less output during training.",
    )
    parser.add_argument(
        "--run-test",
        dest="run_test",
        action="store_true",
        default=True,
        help="Run on testset after training.",
    )
    parser.add_argument(
        "--azureml_logging",
        dest="azureml_logging",
        action="store_true",
        help="Log task results using AML run context.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")

    parser.add_argument(
        "--hyperdrive-arg-parse",
        dest="hyperdrive_arg_parse",
        action="store_true",
        help='Enable hyperdrive argument parsing, in which unknown options "--key val" are interpreted as hyperparameter "key" with value "val".',
    )

    return parser
