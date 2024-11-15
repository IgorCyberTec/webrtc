import asyncio
import threading
import pygame
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

# Conexão WebRTC
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

# Inicializar suporte ao joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Primeiro controle conectado
joystick.init()

# Estados de movimento e ação
is_moving = False  # Indica se o robô está se movendo
is_executing_action = False  # Indica se o robô está executando uma ação
is_action_pending = False  # Impede ações simultâneas
last_command = {"x": 0, "y": 0, "z": 0}  # Último comando enviado

# Fila para comandos assíncronos
command_queue = asyncio.Queue()

# Função para enviar comandos
async def send_command(api_id, parameter=None):
    try:
        print(f"Enviando comando: {api_id}, parâmetros: {parameter}")
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": api_id, "parameter": parameter if parameter else {}}
        )
    except Exception as e:
        print(f"Erro ao enviar comando: {e}")

# Função para mover o dispositivo
async def move_device(x, y, z):
    await send_command(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

# Consumidor da fila de comandos
async def process_command_queue():
    while True:
        command = await command_queue.get()
        await command
        command_queue.task_done()

# Função para redefinir o estado de execução de ação
def reset_is_executing_action():
    global is_executing_action, is_action_pending
    is_executing_action = False
    is_action_pending = False

# Função para enviar comandos de movimento
def send_movement_command(x, y, z):
    """Envia um comando de movimento somente se ele for diferente do último."""
    global last_command
    new_command = {"x": x, "y": y, "z": z}
    # Envia o comando apenas se for diferente do último
    if new_command != last_command:
        asyncio.run_coroutine_threadsafe(
            command_queue.put(move_device(x, y, z)), loop
        )
        last_command = new_command  # Atualiza o último comando enviado

# Função para parar o movimento
def stop_movement():
    """Para o movimento do robô somente se necessário."""
    global is_moving, last_command
    # Verifica se o último comando enviado já era de parar
    if last_command != {"x": 0, "y": 0, "z": 0}:
        send_movement_command(0, 0, 0)  # Envia comando de parar
        is_moving = False

# Função para lidar com eventos do joystick
def handle_joystick_events():
    """Lida com eventos do joystick e controla o movimento do robô."""
    global is_moving, is_executing_action

    axis_forward_back = joystick.get_axis(1)
    axis_left_right = joystick.get_axis(0)
    axis_rotation = joystick.get_axis(3)

    if abs(axis_forward_back) <= 0.2 and abs(axis_left_right) <= 0.2 and abs(axis_rotation) <= 0.2:
        stop_movement()
        return

    if abs(axis_forward_back) > 0.2 and not is_executing_action:
        send_movement_command(-axis_forward_back * 0.5, 0, 0)
        is_moving = True

    if abs(axis_left_right) > 0.2 and not is_executing_action:
        send_movement_command(0, axis_left_right * 0.5, 0)
        is_moving = True

    if abs(axis_rotation) > 0.2 and not is_executing_action:
        send_movement_command(0, 0, axis_rotation * 0.5)
        is_moving = True

    if not is_moving and not is_executing_action:
        if joystick.get_button(0):
            send_action_command(SPORT_CMD["StandUp"])
        elif joystick.get_button(1):
            send_action_command(SPORT_CMD["Sit"])
        elif joystick.get_button(2):
            send_action_command(SPORT_CMD["WiggleHips"])
        elif joystick.get_button(3):
            send_action_command(SPORT_CMD["Dance1"])

def send_action_command(api_id):
    global is_executing_action, is_action_pending
    if not is_action_pending:
        is_executing_action = True
        is_action_pending = True
        asyncio.run_coroutine_threadsafe(
            command_queue.put(send_command(api_id)), loop
        )
        duration = {"StandUp": 3.0, "Sit": 3.0, "WiggleHips": 5.0, "Dance1": 10.0}.get(api_id, 2.0)
        threading.Timer(duration, reset_is_executing_action).start()

def run_asyncio_loop():
    async def setup():
        await conn.connect()
        asyncio.create_task(process_command_queue())
    loop.run_until_complete(setup())
    loop.run_forever()

loop = asyncio.new_event_loop()
asyncio_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
asyncio_thread.start()

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Controlador do Dispositivo")

running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        handle_joystick_events()

finally:
    loop.call_soon_threadsafe(loop.stop)
    asyncio.run(asyncio.sleep(0.1))
    pygame.quit()
