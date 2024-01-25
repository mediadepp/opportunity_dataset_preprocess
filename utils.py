# IHSN 
__author__ = "Mad Medi" 

from functools import wraps
import time 


logger_path = "logger.log"
def logger(*msg): 
    with open(logger_path, mode="a", encoding="utf8") as f: 
        msg = " ".join([str(item) for item in msg]) 
        f.write(f"{msg} \n")


def timer(func): 
    @wraps(func) 
    def wrapper(*args, **kwargs): 
        logger("================================================================================")
        logger("Executing", f"`{func.__name__}`")
        beg = time.time() 
        result = func(*args, **kwargs) 
        end = time.time() 
        logger("Time taken to execute", f"{func.__name__} is {(end-beg)*1e0}")
        logger("================================================================================")
        return result 
    return wrapper 
