import asyncio
import websockets

async def test_connection():
    uri = "ws://158.130.55.26:8000"  # For franka laptop, change localhost to public ip of this machine
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")

asyncio.run(test_connection())
