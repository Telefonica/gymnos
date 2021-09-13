#
#
#   Stable-baselines 3 mixins
#
#

import os
import sys
import mlflow
import numpy as np

from .sb3_mixins_hydra_conf import SaveStrategy

from typing import Any, Dict, Union, Tuple, Optional
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, HumanOutputFormat, KVWriter
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes


class MlflowKVWriter(KVWriter):

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        mlflow.log_metrics(key_values, step=step)

    def close(self):
        pass


class SB3Trainer:

    def create_env(self):
        raise NotImplementedError()

    def create_model(self, env):
        raise NotImplementedError()

    def train(self):
        env = self.create_env()
        model = self.create_model(env)

        assert isinstance(model, BaseAlgorithm)

        self._model_cls = model.__class__

        output_formats = [MlflowKVWriter()]

        if self.verbose:
            output_formats.insert(0, HumanOutputFormat(sys.stdout))

        logger = Logger(folder=None, output_formats=output_formats)

        model.set_logger(logger)

        callbacks = []

        eval_env = self.create_env()

        callback_on_new_best = None
        if self.stop_training_reward_threshold is not None:
            callback_on_new_best = StopTrainingOnRewardThreshold(reward_threshold=self.stop_training_reward_threshold,
                                                                 verbose=self.verbose)

        best_model_save_path = None
        if self.save_strategy == SaveStrategy.BEST:
            best_model_save_path = "saved_models"

        eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path, eval_freq=self.eval_freq,
                                     deterministic=True, render=False, callback_on_new_best=callback_on_new_best,
                                     verbose=self.verbose, n_eval_episodes=self.n_eval_episodes)
        callbacks.append(eval_callback)

        if self.max_num_train_episodes is not None:
            callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=self.max_num_train_episodes,
                                                              verbose=self.verbose)
            callbacks.append(callback_max_episodes)

        try:
            model.learn(self.num_train_timesteps, callback=callbacks, log_interval=self.log_interval)
        finally:
            if self.save_strategy == SaveStrategy.BEST:
                mlflow.log_artifact(os.path.join(eval_callback.best_model_save_path, "best_model.zip"))
            elif self.save_strategy == SaveStrategy.LAST:
                os.makedirs("saved_models", exist_ok=True)
                model.save(os.path.join("saved_models", "last_model"))
                mlflow.log_artifact(os.path.join("saved_models", "last_model.zip"))
            else:
                raise ValueError(f"Unexpected save {self.save_strategy}")

    def test(self):
        test_env = self.create_env()

        save_fname = "best_model" if self.save_strategy == SaveStrategy.BEST else "last_model"
        model = self._model_cls.load(os.path.join("saved_models", save_fname))

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            test_env,
            n_eval_episodes=self.num_test_episodes,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )

        mlflow.log_metrics({
            "test/mean_ep_length": np.mean(episode_lengths),
            "test/mean_reward": np.mean(episode_rewards)
        })


class SB3Predictor:

    def load_model(self, path):
        raise NotImplementedError()

    def load(self, config, run, artifacts_dir):
        if config.trainer.save_strategy == SaveStrategy.BEST.name:
            fname = "best_model.zip"
        elif config.trainer.save_strategy == SaveStrategy.LAST.name:
            fname = "last_model.zip"
        else:
            raise ValueError(f"Unexpected `save_strategy`: {config.trainer.save_strategy}")

        self.model = self.load_model(os.path.join(artifacts_dir, fname))

    def predict(self, obs, state: Optional[np.ndarray] = None, done: Optional[np.ndarray] = None,
                deterministic=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model’s action(s) from an observation

        Parameters
        ----------
        obs:
            Input observation
        state:
            The last states (can be None, used in recurrent policies)
        done:
            The last masks (can be None, used in recurrent policies)
        deterministic:
             Whether or not to return deterministic actions.

        Returns
        -------
        Tuple[ndarray, Optional[ndarray]]
            The model’s action and the next state (used in recurrent policies)
        """
        return self.model.predict(obs, state, done, deterministic=deterministic)
