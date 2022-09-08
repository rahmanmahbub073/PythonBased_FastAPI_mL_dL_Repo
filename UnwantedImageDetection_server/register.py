import json
import aiohttp
import asyncio
import logging
import math


class Register:

    def __init__(self, server_ip, server_port,
                 manager_ip, manager_port,
                 manager_interface_reg, manager_interface_cancel,
                 model_type, model_version):
        self.server_ip = server_ip
        self.server_port = server_port
        self.manager_ip = manager_ip
        self.manager_port = manager_port
        self.manager_interface_reg = manager_interface_reg
        self.manager_interface_cancel = manager_interface_cancel
        self.model_type = model_type
        self.model_version = model_version
        self.status = 0
        self.latest_version = -1
        self.on_success_callback = {}
        self.on_fail_callback = {}
        self.on_success_callback_no_args = []
        self.on_fail_callback_no_args = []

    def on_success(self, func):
        self.on_success_callback_no_args.append(func)

    def on_fail(self, func):
        self.on_fail_callback_no_args.append(func)

    def set_on_success(self, func, *args, **kwargs):
        self.on_success_callback[func] = {'args': args, 'kwargs': kwargs}

    def set_on_fail(self, func, *args, **kwargs):
        self.on_fail_callback[func] = {'args': args, 'kwargs': kwargs}

    async def _check_server(self, times):
        url = f"http://{self.server_ip}:{self.server_port}/health"
        for i in range(times):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response = await response.text()
                        result = json.loads(response)
            except Exception as e:
                logging.warning(f"[register]<self server connect error> exception={e}")
                await asyncio.sleep(0.1 * i)
                continue
            return False if result['code'] != 200 else True
        return False

    def _get_sleep(self, times):
        return 1800 / (1 + math.exp(20 - times))

    async def _register(self):
        start = await self._check_server(10)
        if start:
            logging.info(f"[register]<self server start success>")
        else:
            logging.error(f"[register]<self server start fail>")
            return
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({"port": self.server_port, "model_type": self.model_type, "version": self.model_version})
        url = f"http://{self.manager_ip}:{self.manager_port}{self.manager_interface_reg}"
        times = 0
        while True:
            times += 1
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, data=data) as response:
                        response = await response.text()
                        result = json.loads(response)
            except Exception as e:
                self.status = -1
                logging.error(f"[register]<register server connect error> exception={e} connect_times={times}")
                await asyncio.sleep(self._get_sleep(times))
                continue
            else:
                logging.info(f"[register]<register server connect success>")
                break
        if result['code'] == 200:
            self.status = 1
            self.latest_version = result['latest_version']
            if self.model_version < self.latest_version:
                logging.warning(f"[register]<success, version too old!> \
                                 current_version={self.model_version} latest_version={self.latest_version}")
                self.status = 2
            logging.info(f"[register]<success> latest_version={result['latest_version']}")
        elif result['code'] == 201:
            logging.info(f"[register]<already register>")
            self.status = 1
        else:
            self.status = -1
            logging.error(f"[register]<fail!>")

        if result['code'] in [200, 201]:
            for func in self.on_success_callback:
                args, kwargs = self.on_success_callback[func]['args'], self.on_success_callback[func]['kwargs']
                await func(*args, **kwargs)
            for func in self.on_success_callback_no_args:
                await func()
        else:
            for func in self.on_fail_callback:
                args, kwargs = self.on_fail_callback[func]['args'], self.on_fail_callback[func]['kwargs']
                await func(*args, **kwargs)
            for func in self.on_fail_callback_no_args:
                await func()

    def __call__(self):
        asyncio.create_task(self._register())

    async def cancel(self):
        if self.status == 3:
            return True
        url = f"http://{self.manager_ip}:{self.manager_port}{self.manager_interface_cancel}"
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({"port": self.server_ip, "model_type": self.model_type, "version": self.model_version})
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, data=data) as response:
                    response = await response.text()
                    result = json.loads(response)
        except Exception as e:
            self.status = -1
            logging.info(f"[cancel]<register server connect error> exception={e}")
            return False
        if result['code'] == 200:
            logging.info(f"[cancel]<cancel success>")
            self.status = 3
        else:
            logging.error(f"[cancel]<cancel fail>")
        return True

    def dump(self):
        return {
            'status': self.status,
            'server_ip': self.server_ip,
            'server_port': self.server_port,
            'manager_ip': self.manager_ip,
            'manager_port': self.manager_port,
            'manager_interface_register': self.manager_interface_reg,
            'manager_interface_cancel': self.manager_interface_cancel,
            'model_type': self.model_type,
            'model_version': self.model_version
        }
