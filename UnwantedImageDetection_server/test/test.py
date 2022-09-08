import asyncio
import aiohttp
import json
import base64
import time

async def send(server_url, image, times=1):
    start = time.time()
    for _ in range(times):
        with open(image, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        async with aiohttp.ClientSession() as session:
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({"content": data})
            async with session.post(f"http://{server_url}/model/predict", headers=headers, data=data) as response:
                response = await response.text()
                print(response)
    print('time(ms):', (time.time() - start) * 1000 / times)

async def main():
    start = time.time()
    tasks = []
    for i in range(5000):
        if i % 2 == 0:
            tasks.append(asyncio.create_task(send('127.0.0.1:8000', 'test.jpg')))
        else:
            tasks.append(asyncio.create_task(send('127.0.0.1:5000', 'test.jpg')))
    [await task for task in tasks]
    print('all time(ms):', (time.time() - start) * 1000 / 5000)

if __name__ == "__main__":
    asyncio.run(send('127.0.0.1:9000', 'test.jpg', 10000))
    # asyncio.run(main())