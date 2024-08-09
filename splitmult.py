from decimal import Decimal
import numpy as np
from numba import cuda
import math

@cuda.jit
def add_chunks_kernel(a_chunks, b_chunks, result_chunks, num_chunks_x, num_chunks_y):
    idx = cuda.grid(1)
    if idx < num_chunks_x:
        result_chunks[idx] = a_chunks[idx] + b_chunks[idx]
        if(result_chunks[idx] >= 10.0):
            result_chunks[idx - 1] = result_chunks[idx - 1] + 0.0000001
            result_chunks[idx] = result_chunks[idx] - 10.0

@cuda.jit
def multiply_chunks_kernel(a_chunks, b_chunks, result_chunks, num_chunks_x, num_chunks_y):
    idx = cuda.grid(1)

    if idx < num_chunks_x:
        for i in range(num_chunks_y):
            product = a_chunks[idx] * b_chunks[i]
            scaled_value = product * 10**15
            first_part = int(scaled_value // 10**8)
            second_part = int(scaled_value % 10**8)
            result_chunks[idx + i] += (first_part / (10**7))
            result_chunks[idx + i + 1] += (second_part / (10**7))
        
        for i in range(num_chunks_x + num_chunks_y - 2):
            if result_chunks[i] >= 10.0:
                carry = int(result_chunks[i] // 10.0)
                result_chunks[i] -= carry * 10.0
                if i + 1 < len(result_chunks):
                    cuda.atomic.add(result_chunks, i + 1, carry)
                   
def split_decimal_to_floats(number):
    number_str = str(number).replace('.', '')
    chunk_size = 8
    integer_array = []
    for i in range(0, len(number_str), chunk_size):
        chunk = number_str[i:i + chunk_size]
        integer_array.append(np.float64(chunk) / pow(10, len(chunk) - 1))
    return integer_array

def round_to_significant_digits(value, digits):
    formatted = f"{value:.8f}"
    formatted = formatted.rstrip('0')
    if '.' in formatted:
        integer_part, decimal_part = formatted.split('.')
        decimal_length = len(decimal_part)
        if decimal_length < 7:
            decimal_part = decimal_part.ljust(7, '0')
            formatted = f"{integer_part}.{decimal_part}"
    else:
        # If there is no decimal point, add ".0000000"
        formatted = f"{formatted}.0000000"
    integer_part, decimal_part = formatted.split('.')
    if len(decimal_part) < 7:
        decimal_part = decimal_part.ljust(7, '0')
    return f"{integer_part}.{decimal_part[:7]}"

def remove_trailing_zeros(s):
    i = len(s) - 1
    while i >= 0 and s[i] == '0':
        i -= 1
    
    if i >= 0:
        return s[:i + 1]
    else:
        return s 

def chunks_to_decimal(float_array):
    concatenated_str = ''.join(str(round_to_significant_digits(f, 8)).replace('.', '') for f in float_array)
    combined_value = concatenated_str[0] + '.' + concatenated_str[1:]
    return remove_trailing_zeros(combined_value)

def add(center_x, center_y):
    result_chunks = np.zeros(len(center_x) + len(center_y), dtype=np.float64)
    threads_per_block = 256
    num_chunks_x = len(center_x)
    num_chunks_y = len(center_y)
    blocks_per_grid = (len(result_chunks) + (threads_per_block - 1)) // threads_per_block
    
    add_chunks_kernel[blocks_per_grid, threads_per_block](center_x, center_y, result_chunks, num_chunks_x, num_chunks_y)
    return result_chunks

center_x = Decimal('0.000000000000009')
center_y = Decimal('1.654321012345678')

x_chunks = np.array(split_decimal_to_floats(center_x))
y_chunks = np.array(split_decimal_to_floats(center_y))

results = np.array(add(x_chunks, y_chunks))

print(f"X Chunks: {x_chunks}")
print(f"Y Chunks: {y_chunks}")
print(f"R Chunks: {results}")
print(f"Final Result: {chunks_to_decimal(results)}")