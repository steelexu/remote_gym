# remote simulation
multi-cpu cluster usually don't have GPU. so remote acess is important

## background

for the [prosthetics training](http://osim-rl.stanford.edu/)

## first step
utilize openai [http-api](https://github.com/openai/gym-http-api)


reinforement learning using tensorsource instead of keras-rl

## second step

utilize websocker? from [rl-server](https://github.com/parilo/rl-server) by websocket instead of pyro


this actually let the gpu machine as server, reverse to http-api,,every gym will be client like websocket/rl_client.py, such design is out my expectation, be sure do `pip install websockets websocket-client`


some error with line : req_json = await websocket.recv() ?
