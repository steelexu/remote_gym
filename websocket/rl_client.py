import json
from websocket import create_connection
#https://pypi.python.org/pypi/websocket-client/


class RLClient (object):

    def __init__ (self):

        self.websocket = create_connection('ws://127.0.0.1:8765')

    def act (self, state):
        req = json.dumps({
            'method' : 'act',
            'state' : state
        })
        self.websocket.send(req)

        resp = json.loads(self.websocket.recv())
        print(resp)
        return resp

    def store_exp (self, reward, action, prev_state, next_state):
        req = json.dumps({
            'method' : 'store_exp_batch',
            'rewards' : [reward],
            'actions' : [action],
            'prev_states' : [prev_state],
            'next_states' : [next_state]
        })
        self.websocket.send(req)
        self.websocket.recv()

if __name__ == '__main__':
    rlc=RLClient()
    rlc.act(1)