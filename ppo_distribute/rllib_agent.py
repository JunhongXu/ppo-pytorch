from ray.rllib.agent import Agent
from ray.rllib.optimizers.local_sync import LocalSyncOptimizer
from torch_ppo_evaluator import TorchPPOEvaluator


class TorchPPOAgent(Agent):
    # TODO:
    def _init(self):
        # self.optimizer = LocalSyncOptimizer.make()
        pass

    def compute_action(self, observation):
        pass

    def _save(self, checkpoint_dir=None):
        pass

    def _restore(self, checkpoint_path):
        pass

    def _train(self):
        pass

    @property
    def _agent_name(self):
        return 'Torch-PPO-Agent'

    @property
    def _default_config(self):
        raise NotImplementedError
