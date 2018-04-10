from ray.rllib.agent import Agent
from ray.rllib.optimizers.local_sync import LocalSyncOptimizer
from ray.rllib.optimizers.sample_batch import SampleBatch
from ray.rllib.utils.sampler import SyncSampler
from ppo_distribute.torch_ppo_evaluator import TorchPPOEvaluator
import ray


DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Number of steps after which the rollout gets cut
    "horizon": 2000,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "sgd_stepsize": 5e-5,
    # TODO(pcm): Expose the choice between gpus and cpus
    # as a command line argument.
    "devices": ["/cpu:%d" % i for i in range(4)],
    "tf_session_args": {
        "device_count": {"CPU": 4},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "intra_op_parallelism_threads": 1,
        "inter_op_parallelism_threads": 2,
    },
    # Batch size for policy evaluations for rollouts
    "rollout_batchsize": 1,
    # Total SGD batch size across all devices for SGD
    "sgd_batchsize": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Config params to pass to the model
    "model": {"free_log_std": False},
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch": 4000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "steps_per_rollout": 400,
    # Number of actors used to collect the rollouts
    "num_workers": 8,
    # Resource requirements for remote actors
    "worker_resources": {'num_cpus': None, 'num_gpus': 1},
    # Dump TensorFlow timeline after this many SGD minibatches
    "full_trace_nth_sgd_batch": -1,
    # Whether to profile data loading
    "full_trace_data_load": False,
    # Outer loop iteration index when we drop into the TensorFlow debugger
    "tf_debug_iteration": -1,
    # If this is True, the TensorFlow debugger is invoked if an Inf or NaN
    # is detected
    "tf_debug_inf_or_nan": False,
    # If True, we write tensorflow logs and checkpoints
    "write_logs": True,
    # Arguments to pass to the env creator
    "env_config": {},
    "cuda": True,
    "lr": 5e-5
}


class TorchPPOAgent(Agent):

    # TODO:
    def _init(self):
        # local evaluator for update weights
        self.local_evaluator = TorchPPOEvaluator(env_creator=self.env_creator, config=self.config)
        # create remote evaluators for rollout trajectories; need to add remote to store them to [object store]
        # on this machine
        RemoteEvaluators = ray.remote(**self.config['worker_resources'])(TorchPPOEvaluator)
        # TODO: Need to know why this does not create more than 1 actor???
        self.remote_evaluators = []
        for i in range(self.config['num_workers']):
            self.remote_evaluators.append(RemoteEvaluators.remote(self.config, self.env_creator))

    def compute_action(self, observation):
        pass

    def _save(self, checkpoint_dir=None):
        pass

    def _restore(self, checkpoint_path):
        pass

    def _train(self):
        local_model = self.local_evaluator
        remote_models = self.remote_evaluators

        # sync parameters
        weight_id = ray.put(local_model.get_weights())
        [remote_model.set_weights.remote(weight_id) for remote_model in remote_models]

        sample_batches = collect_samples(remote_models, self.config, self.local_evaluator)

        # TODO: Do PPO update

    @property
    def _agent_name(self):
        return 'Torch-PPO-Agent'

    @property
    def _default_config(self):
        return DEFAULT_CONFIG


# copied from https://github.com/ray-project/ray/blob/master/python/ray/rllib/ppo/rollout.py
def collect_samples(agents, config, local_evaluator):
    num_timesteps_so_far = 0
    trajectories = []
    # This variable maps the object IDs of trajectories that are currently
    # computed to the agent that they are computed on; we start some initial
    # tasks here.

    agent_dict = {}

    for agent in agents:
        fut_sample = agent.sample.remote()
        agent_dict[fut_sample] = agent

    while num_timesteps_so_far < config["timesteps_per_batch"]:
        # TODO(pcm): Make wait support arbitrary iterators and remove the
        # conversion to list here.
        [fut_sample], _ = ray.wait(list(agent_dict))
        agent = agent_dict.pop(fut_sample)
        # Start task with next trajectory and record it in the dictionary.
        fut_sample2 = agent.sample.remote()
        agent_dict[fut_sample2] = agent

        next_sample = ray.get(fut_sample)
        num_timesteps_so_far += next_sample.count
        trajectories.append(next_sample)

    return SampleBatch.concat_samples(trajectories)


if __name__ == '__main__':
    import torch.nn as nn
    ray.init(num_cpus=8, num_gpus=1)

    ppo_agent = TorchPPOAgent(env='MountainCarContinuous-v0')
    ppo_agent._train()



