#!/usr/bin/env python

import asyncio
import websockets
import json
#from rl_train_loop import RLTrainLoop
#from osim_rl_ddpg import OsimRL
# from osim_rl_ddpg_lstm import OsimRLLSTM  # not garanteed to work

num_actions = 18;
observation_size = 41 * 3

#train_loop = RLTrainLoop (num_actions, observation_size)
#osim_rl = OsimRL (train_loop)
# osim_rl = OsimRLLSTM (train_loop)

#train_loop.set_loss_op (osim_rl.get_loss_op ())
#train_loop.add_train_ops (osim_rl.get_train_ops ())
#train_loop.init_vars ()

async def agent_connection(websocket, path):
    while websocket.open:
        req_json = await websocket.recv()
        req = json.loads(req_json)

        method = req ['method']
        if method == 'act':
            #action = osim_rl.act (req ['state'])
            action = json.dumps({
                'musle1' : '35',
                'musle2' : '335'
            })
            await websocket.send(json.dumps(action))
        elif method == 'act_batch':
            actions = osim_rl.act_batch (req ['states'])
            await websocket.send(json.dumps(actions))
        elif method == 'store_exp_batch':
            await websocket.send('')

print("start training")
#train_loop.train ()

start_server = websockets.serve(agent_connection, '0.0.0.0', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
#train_loop.join ()
