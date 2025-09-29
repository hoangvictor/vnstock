from functools import partial
import json
import threading

import websocket


class WSClient:
    def __init__(self, url, handshake_msg: dict):
        self.opened = threading.Event()
        self.done = threading.Event()
        self.messages = []
        self.ws = websocket.WebSocketApp(
            url,
            on_open=partial(self.on_open, handshake_msg=handshake_msg),
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send_message(self, msg: dict):
        if not self.opened.wait(timeout=2):
            raise RuntimeError("WebSocket not open after 10s")
        self.ws.send(json.dumps(msg) + chr(30))

    def on_close(self, ws, close_status_code, close_msg):
        print("Connection closed")

    def on_error(self, ws, error):
        print("Error:", error)

    def on_open(self, ws, handshake_msg: dict):
        ws.send(json.dumps(handshake_msg) + chr(30))
        self.opened.set()

    def on_message(self, ws, message):
        self.messages.append(message)

    def wait_until_done(self, timeout=None):
        self.done.wait(timeout)  # wait until signaled
        return self.messages

    def run(self):
        print("Start connecting to Websocket")
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def send(self, msg):
        self.ws.send(msg)
