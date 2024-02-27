import time

def clock(form = "%H:%M:%S"):
    current_time = time.localtime()
    formatted_time = time.strftime(form, current_time)      
    return formatted_time  