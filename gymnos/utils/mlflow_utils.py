#
#
#   MLFlow utils
#
#

import os
import mlflow
import shutil
import tempfile

from mlflow_export_import.common import filesystem as _filesystem


def strip_underscores(obj):
    return { k[1:]:v for (k,v) in obj.__dict__.items() }


def _mk_local_path(path):
    return path.replace("dbfs:", "/dbfs")


def create_tags_for_metadata(src_client, run, export_metadata_tags):
    """ Create destination tags from source run """
    tags = run.data.tags.copy()
    for k in _databricks_skip_tags:
        tags.pop(k, None)
    if export_metadata_tags:
        uri = mlflow.tracking.get_tracking_uri()
        tags[TAG_PREFIX_METADATA+".tracking_uri"] = uri
        dbx_host = os.environ.get("DATABRICKS_HOST",None)
        if dbx_host is not None:
            tags[TAG_PREFIX_METADATA+".DATABRICKS_HOST"] = dbx_host
        now = int(time.time()+.5)
        snow = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
        tags[TAG_PREFIX_METADATA+".timestamp"] = str(now)
        tags[TAG_PREFIX_METADATA+".timestamp_nice"] = snow
        tags[TAG_PREFIX_METADATA+".user_id"] = run.info.user_id
        tags[TAG_PREFIX_METADATA+".run_id"] =  str(run.info.run_id)
        tags[TAG_PREFIX_METADATA+".experiment_id"] = run.info.experiment_id
        tags[TAG_PREFIX_METADATA+".artifact_uri"] = run.info.artifact_uri
        tags[TAG_PREFIX_METADATA+".status"] = run.info.status
        tags[TAG_PREFIX_METADATA+".lifecycle_stage"] = run.info.lifecycle_stage
        tags[TAG_PREFIX_METADATA+".start_time"] = run.info.start_time
        tags[TAG_PREFIX_METADATA+".end_time"] = run.info.end_time
        #tags[TAG_PREFIX_METADATA+".status"] = run.info.status
        exp = src_client.get_experiment(run.info.experiment_id)
        tags[TAG_PREFIX_METADATA+".experiment_name"] = exp.name
    tags = { k:v for k,v in sorted(tags.items()) }
    return tags


class MLFlowExporter:
    """
    Adapted from https://github.com/amesar/mlflow-export-import
    """

    def __init__(self, client=None, export_metadata_tags=False, filesystem=None):
        self.client = client or mlflow.tracking.MlflowClient()
        self.fs = filesystem or _filesystem.get_filesystem()
        print("Filesystem:", type(self.fs).__name__)
        self.export_metadata_tags = export_metadata_tags

    def export_run(self, run_id, output):
        run = self.client.get_run(run_id)
        if output.endswith(".zip"):
            return self.export_run_to_zip(run, output)
        else:
            self.fs.mkdirs(output)
            return self.export_run_to_dir(run, output)

    def export_run_to_zip(self, run, zip_file):
        temp_dir = tempfile.mkdtemp()
        try:
            self.export_run_to_dir(run, temp_dir)
            utils.zip_directory(zip_file, temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def export_run_to_dir(self, run, run_dir):
        tags = utils.create_tags_for_metadata(self.client, run, self.export_metadata_tags)
        dct = {"info": strip_underscores(run.info),
               "params": run.data.params,
               "metrics": run.data.metrics,
               "tags": tags,
               }
        path = os.path.join(run_dir, "run.json")
        utils.write_json_file(self.fs, path, dct)

        # copy artifacts
        dst_path = os.path.join(run_dir, "artifacts")
        artifacts = self.client.list_artifacts(run.info.run_id)
        if len(artifacts) > 0:  # Because of https://github.com/mlflow/mlflow/issues/2839
            self.fs.mkdirs(dst_path)
            self.client.download_artifacts(run.info.run_id, "", dst_path=_mk_local_path(dst_path))
