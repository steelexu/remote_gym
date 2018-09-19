# remote simulation
multi-cpu cluster usually don't have GPU. so remote acess is important

## background

for the [prosthetics training](http://osim-rl.stanford.edu/)

## first step
utilize openai [http-api](https://github.com/openai/gym-http-api)


reinforement learning using tensorsource instead of keras-rl

### with keras-rl

only tested with Pendulum-v0

local_env as wrapper of http-client

http-server modification
* add item() on  ndarray for jsonify
* action need to be array [] instead of float, now only support 1 input :(
* how to add seed in create call??


### modification of keras-rl

* action_space.sample() changed to action_space_sample, core.py
* disable the client render, Visualizer.on_action_end
 /home/fish/anaconda3/envs/opensim-rl/lib/python3.6/site-packages/keras_rl-0.4.2-py3.6.egg/rl/callbacks.py


## second step

utilize websocker? from [rl-server](https://github.com/parilo/rl-server) by websocket instead of pyro


this actually let the gpu machine as server, reverse to http-api,,every gym will be client like websocket/rl_client.py, such design is out my expectation, be sure do `pip install websockets websocket-client`


some error with line : req_json = await websocket.recv() ?
