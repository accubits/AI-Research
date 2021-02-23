import paho.mqtt.client as mqtt
from src.gpt_train import train as gpt_trainer

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("filename")

def on_message(client, userdata, msg):
    filename = msg.payload.decode()
    gpt_trainer(filename)
    # client.disconnect()
    

if __name__ == "__main__":
    client = mqtt.Client()
    client.connect("localhost")

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()