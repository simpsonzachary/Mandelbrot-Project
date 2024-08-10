from decimal import Decimal
import numpy as np
from numba import cuda
import math

@cuda.jit
def add_chunks_kernel(a, b, result, digits):
    if cuda.grid(1) == 0:
        if(a[0] == b[0]):
            result[0] = a[0]
            for i in range(digits, 0, -1):
                result[i] += a[i] + b[i]
                if(result[i] >= 10):
                    result[i - 1] += 1
                    result[i] -= 10
                    
        if (a[0] == 1 and b[0] == 0):
            for i in range(digits, 0, -1):
                result[i] += b[i] - a[i]
                if(result[i] < 0):
                    result[i - 1] -= 1
                    result[i] += 10
            # for i in range(1, digits, 1):
            #     if(result[i] != 0):
            #         if(result[i] <0):
            #             result[i] *= -1
            #             result[0] = 1
        
def to_chunks(number, num_digits):
    number_str = number.replace('.', '')
    if number_str[0] == '-':
        number_str = number_str.replace('-', '1')
    else:
        number_str = '0' + number_str[0:]
    required_length = num_digits + 1
    
    # Pad number_str with zeros if it's shorter than required_length
    if len(number_str) < required_length:
        number_str = number_str.ljust(required_length, '0')
    else:
        # Truncate number_str to required_length if it's longer
        number_str = number_str[:required_length]
    
    int_array = []
    for i in range(0, required_length):
        num = int(number_str[i])
        int_array.append(num)
    
    return int_array

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

def chunks_to_decimal(arr):
    digits = ''.join(str(digit) for digit in arr)
    number_str = digits[1] + '.' + digits[2:]
    number_str = remove_trailing_zeros(number_str)
    if digits[0] == '0':
        return number_str
    else:
        return '-' + number_str

def add(num1, num2, digits):
    result = np.zeros(digits + 1, dtype=np.int32)
    threads_per_block = 256
    blocks_per_grid = (len(result) + (threads_per_block - 1)) // threads_per_block
    
    add_chunks_kernel[blocks_per_grid, threads_per_block](num1, num2, result, digits)
    return result
    
def update_bounds(min_x, max_x, min_y, max_y, center_x, center_y, zoom):
    range_x = max_x - min_x
    range_y = max_y - min_y

    new_range_x = range_x * zoom
    new_range_y = range_y * zoom

    min_x_new = center_x - new_range_x / 2
    max_x_new = center_x + new_range_x / 2
    min_y_new = center_y - new_range_y / 2
    max_y_new = center_y + new_range_y / 2

    return min_x_new, max_x_new, min_y_new, max_y_new

# main operating portion

num_digits = 4
center_x = '-0.123'
center_y = '0.321'
zoom     = '0.009'

a = np.array(to_chunks(center_x, num_digits))
b = np.array(to_chunks(center_y, num_digits))

result = add(a, b, num_digits)

print(result)