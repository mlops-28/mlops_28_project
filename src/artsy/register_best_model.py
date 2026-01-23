import logging
import operator
import os

from dotenv import load_dotenv
import hydra
import wandb

from artsy import _PATH_CONFIGS

logger = logging.getLogger(__name__)
load_dotenv()


@hydra.main(config_path=_PATH_CONFIGS, config_name="default_config.yaml", version_base=None)
def stage_best_model_to_registry(cfg) -> None:
    """
    Stage the best model to the model registry.

    Args:
        cfg: Configuration file containing the following parameters:
            model_name: Name of the model to be registered.
            metric_name: Name of the metric to choose the best model from.
            higher_is_better: Whether higher metric values are better.

    """
    model_name = cfg.registry.model_name
    metric_name = cfg.registry.metric_name
    higher_is_better = cfg.registry.higher_is_better

    # Initialize API for artifact collection
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact_collection = api.artifact_collection(type_name="model", name=model_name)

    # Search for best model based on metric
    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logger.error("No model found in registry.")
        return

    # Stage best model in model registry
    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=f"{cfg.eval.model_registry}/{cfg.eval.collection}:{model_name}",
        aliases=cfg.registry.staged_model_tags,
    )
    best_artifact.save()
    logger.info("Model staged to registry.")


if __name__ == "__main__":
    print("Running register_best_model.py")
    stage_best_model_to_registry()
