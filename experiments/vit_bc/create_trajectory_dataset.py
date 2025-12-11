import argparse
import os
import torch
import thinker
from tqdm import tqdm
import gym_sokoban
import gym
from gym.vector.utils import batch_space
from thinker.main import Env
from thinker.actor_net import DRCNet
from thinker import util
from typing import NamedTuple
from bc_dataset import BCDataset


@torch.no_grad()
def create_trajectory_dataset(net: DRCNet, env: Env, flags: NamedTuple, num_levels: int, device, starting_level: int) -> list:
    """Generate a list where each entry is a the optimal action given a state (state, action)

    Args:
        net (DRCNet): Trained DRC network used to generate transitions
        env (Env): Sokoban environment
        flags (NamedTuple): flag object
        num_episodes (int): number of episodes to run to generate the transitions
        device (_type_): device to run the model on

    Returns:
        list: returns trajectory_data, a list of tuples (state, action)
    """
    
    trajectory_data = []
    for level_id in tqdm(range(starting_level, starting_level + num_levels)):

        state = env.reset(room_id=level_id)
        state = {'real_states': torch.tensor(state, device=device, dtype=torch.uint8).unsqueeze(0)}
        done = False

        rnn_state = net.initial_state(batch_size=1, device=device)
        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

        actor_out, rnn_state = net(env_out, rnn_state, greedy=True)

        while not done:
            action = actor_out.action

            state = state['real_states'][0].detach().cpu()
            trajectory_data.append([state, action.item()])

            state, reward, done, info = env.step(action)
            state = {'real_states': torch.tensor(state, device=device, dtype=torch.uint8).unsqueeze(0)}
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = torch.tensor([done], device=device, dtype=torch.bool)
            info = {k: torch.tensor([v], device=device) for k, v in info.items()}

            env_out = util.create_env_out(action, state, reward, done, info, flags)
            actor_out, rnn_state = net(env_out, rnn_state, greedy=True)

    return trajectory_data


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="create trajectory dataset")
    parser.add_argument("--starting_level", type=int, default=0, help="number of the level to start from")
    parser.add_argument("--num_levels", type=int, default=900000, help="number of levels in the environment")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--env_name", type=str, default="", help="level dataset to collect transitions from")
    parser.add_argument("--name", type=str, default="train", help="name of dataset to create")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs")
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--only_solved', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 

    env_name_part = f"-{args.env_name}-" if args.env_name else "-"
    env = gym.make(f"Sokoban{env_name_part}v0")
    env = thinker.wrapper.TransposeWrap(env)

    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = True
    flags.mini_unqtar = False

    net = DRCNet(
        obs_space=gym.spaces.Dict({
            "real_states": batch_space(env.observation_space),
        }),
        action_space=batch_space(env.action_space),
        flags=flags,
        record_state=True,
        num_ticks=args.num_ticks,
        num_layers=args.num_layers
    )
    ckp_path = "../../checkpoints/sokoban"
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{args.model_name}.tar")


    ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    net.to(device)
    net.eval()

    trajectory_data = create_trajectory_dataset(
                                        net=net,
                                        env=env,
                                        flags=flags,
                                        num_levels=args.num_levels,
                                        starting_level=args.starting_level,
                                        device=device
                                       )

    print(f"Dataset {args.env_name}_trajectories_{args.model_name} contains {len(trajectory_data)} transitions")
    trajectory_data = list(zip(*trajectory_data))
    
    if not os.path.exists("./data"):
        os.mkdir("./data")
    torch.save(BCDataset(*trajectory_data), f"./data/{args.env_name}_trajectories_{args.starting_level}" + ("_resnet" if args.resnet else "") + ".pt")
