import logging
import argparse
import importlib
import json


from gym_http_client import Client

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.contrib.openai_gym import OpenAIGym

# Create a Proximal Policy Optimization agent

# atari needed
# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000


class RandomDiscreteAgent(object):
    def __init__(self, n):
        self.n = n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    args = parser.parse_args()

    
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states={'shape': (4,), 'type': 'float'},
            actions={'type': 'int', 'num_actions': 2},
            network=network,
        )
    )
    
    
    
    # Set up client
    remote_base = 'http://172.17.0.1:5000'
    client = Client(remote_base)

    # Set up environment
    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)

    # Set up agent
    action_space_info = client.env_action_space_info(instance_id)
    #agent = RandomDiscreteAgent(action_space_info['n'])

    # Run experiment, with monitor
    outdir = '/tmp/random-agent-results'
    client.env_monitor_start(instance_id, outdir, force=True, resume=False, video_callable=False)
    
    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = client.env_reset(instance_id)
        #print("ob:",ob)

        for j in range(max_steps):
            #action = client.env_action_space_sample(instance_id)
            actionx = agent.act(states=OpenAIGym.flatten_state(state=ob))
            #print(type(actionx),actionx)
            actiont = OpenAIGym.unflatten_action(action=actionx)
            #print(type(actiont),actiont)
            ob, reward, done, _ = client.env_step(instance_id, actiont.item(), render=True)
            if done:
                print("done")
                break

    # Dump result info to disk
    client.env_monitor_close(instance_id)

    # Upload to the scoreboard. This expects the 'OPENAI_GYM_API_KEY'
    # environment variable to be set on the client side.
    logger.info("""Successfully ran example agent using
        gym_http_client. Now trying to upload results to the
        scoreboard. If this fails, you likely need to set
        os.environ['OPENAI_GYM_API_KEY']=<your_api_key>""")

    #client.upload(outdir)
