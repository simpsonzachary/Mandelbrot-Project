import numpy as np
from numba import cuda

MAX_DIGITS = 24
MAX_DIGITS_TEMP = MAX_DIGITS * 2
def to_chunks(number):
    number_str = number.replace('.', '')
    if number_str[0] == '-':
        number_str = number_str.replace('-', '1')
    else:
        number_str = '0' + number_str[0:]
    required_length = MAX_DIGITS + 1

    if len(number_str) > required_length:
        # Truncate number_str to required_length if it's longer
        number_str = number_str[:required_length]
    
    int_array = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
    for i in range(len(number_str)):
        num = int(number_str[i])
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

@cuda.jit
def multiply_chunks_kernel(a, b, result):
    temp = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    z0 = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    z1 = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    z2 = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    temp_sum = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    
    m = MAX_DIGITS_TEMP // 2 + 1
    
    for i in range(m):
        for j in range(m):
            z0[i + j] += a[i + 1] * b[j + 1]
    
    for i in range(m):
        for j in range(m):
            z2[i + j] += a[m + i + 1] * b[m + j + 1]
    
    for i in range(m):
        for j in range(m):
            temp_sum[i + j] += (a[i + 1] + a[m + i + 1]) * (b[j + 1] + b[m + j + 1])
    
    for i in range(MAX_DIGITS_TEMP):
        z1[i] = temp_sum[i] - z0[i] - z2[i]
    
    for i in range(0, MAX_DIGITS_TEMP):
        temp[i] = z0[i] + z1[i] * 10**m + z2[i] * 10**(2 * m)
    
    for i in range(MAX_DIGITS_TEMP, 0, -1):
            if(temp[i] >= 10):
                temp[i - 1] += temp[i] / 10
                temp[i] %= 10
                
    result[0] = 0 if (a[0] == b[0]) else 1
    for i in range(0, MAX_DIGITS):
            result[i + 1] = temp[i]
            
def multiply(a, b):
    num1 = to_chunks(a)
    num2 = to_chunks(b)
    result = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
    
    # Launch kernel with one block and one thread
    multiply_chunks_kernel[1, 1](num1, num2, result)
    multiply_chunks_kernel[1, 1](result, num2, result)
    return result

a = '-4.12345678'
b = '4.12345678'

print(multiply(a, b))
print(chunks_to_decimal(multiply(a, b)))
