""" Script to export matplotlib plots from training to tikz """

from hierarchical_policy.decision_maker.ppo_decision_maker import PPO
from hierarchical_policy.updraft_exploiter import model_updraft_exploiter
from policy_evaluation import run_episode
from hierarchical_policy.vertex_tracker.waypoint_controller import ControllerWrapper
from hierarchical_policy.decision_maker import params_decision_maker
import torch
from glider.envs.glider_env_3D import GliderEnv3D

device = torch.device('cpu')
# env = gym.make('glider3D-v0', agent='decision_maker')
env = GliderEnv3D(agent='decision_maker')

# set seed to fix updraft distribution and trajectory
#env.seed(42)
#np.random.seed(42)

waypoint_controller = ControllerWrapper(env)
updraft_exploiter = model_updraft_exploiter.UpdraftExploiterActorCritic().to(device)
updraft_exploiter.load_state_dict(torch.load(
    "../resources/results_paper/policies/updraft_exploiter_actor_critic_final_17-October-2021_20-21.pt", map_location=torch.device('cpu')))
ppo = PPO(waypoint_controller, updraft_exploiter, env)
ppo.model.actor.load_state_dict(torch.load(
    "../resources/results_paper/policies/decision_maker_actor_final_30-October-2021_11-02.pt", map_location=torch.device('cpu')))
_params_agent = params_decision_maker.AgentParameters()

iterations = 10
for plot_number in range(0, iterations):
    print("Running iteration number {}!".format(plot_number))
    run_episode.main(env, ppo, plot_number, _params_agent, validation_mask=True)