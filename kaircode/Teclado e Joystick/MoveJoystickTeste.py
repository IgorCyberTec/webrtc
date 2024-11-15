import asyncio
import threading
import pygame
import cv2
import numpy as np
from time import time
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

# Configuração da conexão WebRTC
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

# Inicializar suporte ao joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Primeiro controle conectado
joystick.init()

# Estados
movement_active = {"forward": False, "backward": False, "rotate_left": False, "rotate_right": False}
movement_tasks = {}
is_moving = False  # Indica se o robô está se movendo
is_executing_action = False  # Indica se o robô está executando uma ação

# Configuração para a câmera
video_feed = None  # Variável para armazenar o feed da câmera
frame_ready = threading.Event()  # Evento para sincronizar o recebimento de frames

# Função para processar os frames da câmera
async def start_video_feed():
    global video_feed

    def on_frame(frame):
        global video_feed
        # Converte o frame recebido para uma imagem OpenCV
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        video_feed = frame
        frame_ready.set()  # Notifica que o frame está pronto

    # Configure o callback para frames de vídeo
    conn.video.on_frame = on_frame  # Substitua isso pelo método correto para registrar o callback de vídeo

# Função para exibir o feed da câmera
def display_camera_feed():
    global video_feed

    while True:
        # Aguarda um novo frame
        frame_ready.wait()
        frame_ready.clear()

        if video_feed is not None:
            cv2.imshow("Camera do Robô", video_feed)

        # Verifica se a janela foi fechada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Função para enviar comandos
async def send_command(api_id, parameter=None):
    print(f"Enviando comando: {api_id}, com parâmetros: {parameter}")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": api_id, "parameter": parameter if parameter else {}}
    )

# Função de movimento contínuo
async def move_device(x, y, z):
    await send_command(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

# Função de redefinir ação
def reset_is_executing_action():
    global is_executing_action
    is_executing_action = False
    print("Ação concluída. Pronto para o próximo comando.")

# Função para enviar comandos de ação
def send_action_command(api_id):
    global is_executing_action
    if is_executing_action:
        print("Já existe uma ação em execução. Aguarde antes de enviar outro comando.")
        return
    is_executing_action = True
    asyncio.run_coroutine_threadsafe(send_command(api_id), loop)
    threading.Timer(1.0, reset_is_executing_action).start()

# Função para lidar com eventos do joystick
def handle_joystick_events():
    global movement_active, movement_tasks, is_moving, is_executing_action

    # Movimentos contínuos baseados nos eixos analógicos
    axis_forward = joystick.get_axis(1)  # Eixo Y do analógico esquerdo
    axis_rotation = joystick.get_axis(0)  # Eixo X do analógico esquerdo

    # Movimento para frente
    if axis_forward < -0.1:
        asyncio.run_coroutine_threadsafe(move_device(x=0.5, y=0, z=0), loop)
    elif axis_forward > 0.1:
        asyncio.run_coroutine_threadsafe(move_device(x=-0.5, y=0, z=0), loop)

    # Rotação para esquerda/direita
    if axis_rotation < -0.1:
        asyncio.run_coroutine_threadsafe(move_device(x=0, y=0, z=0.2), loop)
    elif axis_rotation > 0.1:
        asyncio.run_coroutine_threadsafe(move_device(x=0, y=0, z=-0.2), loop)

    # Ações específicas para botões
    if joystick.get_button(0):  # Botão A
        send_action_command(SPORT_CMD["StandUp"])
    if joystick.get_button(1):  # Botão B
        send_action_command(SPORT_CMD["Sit"])

# Loop asyncio
def run_asyncio_loop():
    async def setup():
        await conn.connect()
        await start_video_feed()

    loop.run_until_complete(setup())
    loop.run_forever()

loop = asyncio.new_event_loop()
asyncio_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
asyncio_thread.start()

# Thread para exibir o feed da câmera
camera_thread = threading.Thread(target=display_camera_feed, daemon=True)
camera_thread.start()

# Inicialização do Pygame
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Controlador do Dispositivo")

# Loop principal
running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Lidar com eventos do joystick
        handle_joystick_events()

finally:
    # Encerrando o loop asyncio, OpenCV e Pygame
    loop.call_soon_threadsafe(loop.stop)
    async def wait_for_loop():
        await asyncio.sleep(1)
    asyncio.run(wait_for_loop())
    pygame.quit()
    cv2.destroyAllWindows()
