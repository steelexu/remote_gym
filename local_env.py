from gym_http_client import Client

class local_env(object):
    def __init__(self, env_id):
        remote_base = 'http://127.0.0.1:5000'
        self.client = Client(remote_base)
        self.instance_id = self.client.env_create(env_id)

    def reset(self):
        obs = self.client.env_reset(self.instance_id)
        return obs

    def step(self,action):
        #print("---sssss---------",type(action),action.item())
        [observation, reward, done, info] = self.client.env_step(self.instance_id,action.item(), True)
        return observation, reward, done, info

    def action_space_sample(self):
        action = self.client.env_action_space_sample(self.instance_id)
        return action

    def action_space_info(self):
        info = self.client.env_action_space_info(self.instance_id)
        return info

    def observation_space_info(self):
        info = self.client.env_observation_space_info(self.instance_id)
        return info

if __name__ == '__main__':
    ENV_NAME = 'Pendulum-v0'
    env= local_env(ENV_NAME)