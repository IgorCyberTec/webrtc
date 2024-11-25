import cv2
import numpy as np
import torch
import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
import speech_recognition as sr

# Configuração do YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 pequeno para testes

# Configuração do logging
logging.basicConfig(level=logging.FATAL)

# Conexão com o robô
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
frame_queue = Queue()

async def recv_camera_stream(track: MediaStreamTrack):
    """
    Função assíncrona para receber frames da câmera do robô.
    """
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)

def setup_camera():
    """
    Configura a câmera do robô e inicia o recebimento de frames.
    """
    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Erro na conexão da câmera: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

async def send_command(api_id, parameter=None):
    """
    Envia comandos ao robô.
    """
    try:
        await conn.datachannel.pub_sub.publish_request_new(
            "SPORT_MOD",
            {"api_id": api_id, "parameter": parameter if parameter else {}}
        )
    except Exception as e:
        print(f"Erro ao enviar comando: {e}")

async def move_robot_towards(dx, dy):
    """
    Move o robô na direção do objeto.
    """
    try:
        await send_command(1008, {"x": dx, "y": dy, "z": 0})
    except Exception as e:
        print(f"Erro ao mover o robô: {e}")

def detect_objects(frame):
    """
    Detecta objetos no frame usando YOLO.
    """
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Coordenadas de detecção
    return detections, results.names

def recognize_speech():
    """
    Usa reconhecimento de fala para capturar comandos.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Diga o nome do objeto:")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio, language="pt-BR")
            print(f"Você disse: {command}")
            return command
        except sr.UnknownValueError:
            print("Não entendi o comando.")
            return None
        except sr.RequestError as e:
            print(f"Erro no serviço de reconhecimento de fala: {e}")
            return None

def find_object(command, detections, class_names, frame_width, frame_height):
    """
    Encontra o objeto solicitado no comando de voz.
    """
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        obj_name = class_names[int(cls)]
        if command.lower() in obj_name.lower():
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dx = (center_x - frame_width / 2) / frame_width  # Normalizado
            dy = (center_y - frame_height / 2) / frame_height
            print(f"Objeto {obj_name} encontrado em ({dx:.2f}, {dy:.2f})")
            return dx, dy
    print(f"Objeto {command} não encontrado.")
    return None, None

async def main():
    setup_camera()  # Configura a câmera do robô

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_height, frame_width, _ = frame.shape

                # Detecta objetos
                detections, class_names = detect_objects(frame)

                # Desenha os objetos detectados no frame
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    obj_name = class_names[int(cls)]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, obj_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Exibe o frame na janela
                cv2.imshow("Câmera do Robô", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Reconhecimento de fala
                command = recognize_speech()
                if command:
                    dx, dy = find_object(command, detections, class_names, frame_width, frame_height)
                    if dx is not None and dy is not None:
                        await move_robot_towards(dx, dy)
            else:
                time.sleep(0.01)

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
