import time
import functools


def timer(name: str = "Function"):
    """
    Decorator to measure execution time of a function.

    Usage:
        @timer("My Task")
        def my_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"\n⏱️  Starting: {name}")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"✅ Finished: {name} in {elapsed:.2f} seconds\n")
            
            return result
        return wrapper
    return decorator