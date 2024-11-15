import asyncio
import threading
import pygame
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

async def move_device(x, y, z):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z}
        }
    )

async def stand_device():
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandOut"], "parameter": {"data": True}}
    )

async def sit_device():
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandOut"], "parameter": {"data": False}}
    )

def handle_key_press(key):
    if key == pygame.K_w:
        asyncio.run_coroutine_threadsafe(move_device(0.2, 0, 0), loop)
    if key == pygame.K_s:
        asyncio.run_coroutine_threadsafe(move_device(-0.2, 0, 0), loop)
    if key == pygame.K_a:
        asyncio.run_coroutine_threadsafe(move_device(0, -0.2, 0), loop)
    if key == pygame.K_d:
        asyncio.run_coroutine_threadsafe(move_device(0, 0.2, 0), loop)

def run_asyncio_loop():
    async def setup():
        await conn.connect()
    loop.run_until_complete(setup())
    loop.run_forever()

loop = asyncio.new_event_loop()
asyncio_thread = threading.Thread(target=run_asyncio_loop)
asyncio_thread.start()

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Controlador do Dispositivo")

running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break  # Sair do loop for

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False  # Sair do loop while
                    break  # Sair do loop for

        # Verificar se uma tecla está sendo pressionada constantemente
        keys = pygame.key.get_pressed()  # Retorna o estado atual das teclas
        if keys[pygame.K_w]:  # Se 'W' estiver pressionada
            handle_key_press(pygame.K_w)
finally:
    loop.call_soon_threadsafe(loop.stop)
    async def wait_for_loop():
        await asyncio.sleep(0.1)  # Espera loop fechar
    asyncio.run(wait_for_loop())
    pygame.quit()
