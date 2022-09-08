import os
from dotenv import load_dotenv
from configparser import ConfigParser

conf = ConfigParser()
conf.read('model.conf')

load_dotenv('.env')

def _getenv(key, default): return type(default)(os.getenv(key)) if os.getenv(key) else default

SERVER_IP = _getenv('SERVER_IP', '0.0.0.0')  # Service IP
SERVER_PORT = _getenv('SERVER_PORT', '6002')  # Service IP

REGISTER = _getenv('REGISTER', 0) # register to the management service

MANAGER_IP = _getenv('MANAGER_IP', '127.0.0.1') # Management server address
MANAGER_PORT = _getenv('MANAGER_PORT', 5005) # Management server address
MANAGER_INTERFACE_REGISTER = _getenv('MANAGER_INTERFACE_REGISTER', '/model/register')
MANAGER_INTERFACE_CANCEL = _getenv('MANAGER_INTERFACE_CANCEL', '/model/cancel')

MODEL_TYPE = _getenv('MODEL_TYPE', conf.get('model', 'model_type', fallback='')) # Service type
MODEL_VERSION = _getenv('MODEL_VERSION', 1) # Service version number

ENGINE_FILE_PATH = _getenv('ENGINE_FILE_PATH', conf.get('model', 'engine_file_path', fallback=''))
CLASS_NUM = _getenv('CLASS_NUM', int(conf.get('model', 'class_num', fallback='0')))
CLASS_NAMES = [name.strip() for name in _getenv('CLASS_NAMES', conf.get('model', 'class_names')).split(',')]

KEY = _getenv('KEY', 'LONGYUAN')