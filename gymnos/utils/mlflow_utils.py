#
#
#   MLFlow utils
#
#

def _strip_underscores(obj):
    return {k[1:]: v for (k, v) in obj.__dict__.items()}


def _get_run_export_tags(run):
    skip_tags = {
        "mlflow.user",
        "mlflow.log-model.history"
    }
    tags = run.data.tags.copy()

    for k in skip_tags:
        tags.pop(k, None)

    tags = {k: v for k, v in sorted(tags.items())}
    return tags


def jsonify_mlflow_run(mlflow_run):
    return {
        "info": _strip_underscores(mlflow_run.info),
        "params": mlflow_run.data.params,
        "metrics": mlflow_run.data.metrics,
        "tags": _get_run_export_tags(mlflow_run),
    }
