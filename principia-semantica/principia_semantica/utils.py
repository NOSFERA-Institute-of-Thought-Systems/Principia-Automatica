# principia_semantica/utils.py
import time
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Executing '{func.__name__}'...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"'{func.__name__}' finished in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper