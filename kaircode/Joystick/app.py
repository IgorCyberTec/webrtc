from flask import Flask, render_template, Response, request, jsonify
import threading
import asyncio
from go2_webrtc_driver.constants import RTC_TOPIC
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from queue import Queue
import cv2
import numpy as np
import json
import subprocess
import os

app = Flask(__name__)

# Configuração da conexão WebRTC
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

# Loop de eventos do asyncio
loop = asyncio.new_event_loop()
asyncio_thread = threading.Thread(target=lambda: loop.run_forever())
asyncio_thread.start()

# Variáveis para armazenar os processos dos scripts
recon_process = None
joystick_process = None  # Processo para o MoveJoystickAvancado.py

# Variável para armazenar o frame de vídeo atual
frame_queue = Queue()

# Função para enviar comandos
async def send_command(api_id, parameter=None):
    print(f"Enviando comando: {api_id}, com parâmetros: {parameter}")
    try:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": api_id, "parameter": parameter if parameter else {}}
        )
    except Exception as e:
        print(f"Erro ao enviar comando: {e}")

def send_move_command(x, y, z):
    asyncio.run_coroutine_threadsafe(
        send_command(1008, {"x": x, "y": y, "z": z}), loop
    )

def send_action_command(api_id):
    asyncio.run_coroutine_threadsafe(
        send_command(api_id), loop
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect')
def connect():
    asyncio.run_coroutine_threadsafe(connect_robot(), loop)
    return jsonify({"status": "Conectado"})

async def connect_robot():
    try:
        await conn.connect()
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001}
        )
        if response['data']['header']['status']['code'] == 0:
            data = json.loads(response['data']['data'])
            current_motion_switcher_mode = data['name']
            print(f"Modo de movimento atual: {current_motion_switcher_mode}")
        if current_motion_switcher_mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}}
            )
    except Exception as e:
        print(f"Erro na conexão: {e}")

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    x = data.get('x', 0)
    y = data.get('y', 0)
    z = data.get('z', 0)
    send_move_command(x, y, z)
    return jsonify({"status": "Movendo", "x": x, "y": y, "z": z})

@app.route('/action', methods=['POST'])
def action():
    data = request.json
    api_id = data.get('api_id')
    send_action_command(api_id)
    return jsonify({"status": "Ação Executada", "api_id": api_id})

@app.route('/toggle_recon', methods=['POST'])
def toggle_recon():
    global recon_process
    if recon_process is None or recon_process.poll() is not None:
        script_path = os.path.abspath("Recon.py")
        recon_process = subprocess.Popen(["python3", script_path])
        status = "Reconhecimento de Gestos Ativado"
    else:
        recon_process.terminate()
        recon_process.wait()
        recon_process = None
        status = "Reconhecimento de Gestos Desativado"
    return jsonify({"status": status})

@app.route('/toggle_joystick', methods=['POST'])
def toggle_joystick():
    global joystick_process
    if joystick_process is None or joystick_process.poll() is not None:
        script_path = os.path.abspath("kaircode/Teclado e Joystick/MoveJoystickAvancado.py")
        joystick_process = subprocess.Popen(["python3", script_path])
        status = "Controle Xbox Ativado"
    else:
        joystick_process.terminate()
        joystick_process.wait()
        joystick_process = None
        status = "Controle Xbox Desativado"
    return jsonify({"status": status})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def setup_video_stream():
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    async def setup():
        try:
            await conn.connect()
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
        except Exception as e:
            print(f"Erro ao configurar o vídeo: {e}")

    asyncio.run_coroutine_threadsafe(setup(), loop)

setup_video_stream()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
