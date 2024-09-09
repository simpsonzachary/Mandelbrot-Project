import numpy as np

MAX_DIGITS = 24 # must be divisible by 3
MAX_DIGITS_TEMP = MAX_DIGITS * 2
TOOM3_CHUNK_DIGITS = MAX_DIGITS // 3

def to_chunks(number):
    number_str = number.replace('.', '')
    if number_str[0] == '-':
        number_str = number_str.replace('-', '1')
    else:
        number_str = '0' + number_str[0:]
    required_length = MAX_DIGITS + 1

    if len(number_str) > required_length:
        number_str = number_str[:required_length]
    
    int_array = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
    for i in range(0, len(number_str)):
        num = np.int32(number_str[i])
        int_array[i] = num
    return int_array

def chunks_to_decimal(arr):
    val = 0.0
    for i in range(1, len(arr)):
        if(arr[i] != 0):
            val += np.float64(np.float64(arr[i]) / 10**(i - 1))
    if arr[0] == 1:
        val *= -1
    return val