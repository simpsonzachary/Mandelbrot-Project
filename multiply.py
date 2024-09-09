import numpy as np
from numba import cuda

from utils import to_chunks, chunks_to_decimal, MAX_DIGITS, MAX_DIGITS_TEMP

@cuda.jit
def multiply_chunks_kernel(a, b, result):
    temp = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int32)
    if (a[0] == b[0]):
        result[0] = 0
    else:
        result[0] = 1
        
    for i in range(0, MAX_DIGITS):
        for j in range(0, MAX_DIGITS):
            temp[i + j] += a[i + 1] * b[j + 1]
        
    for i in range(MAX_DIGITS_TEMP, 0, -1):
        if(temp[i] >= 10):
            temp[i - 1] += temp[i] / 10
            temp[i] %= 10
                
    for i in range(0, MAX_DIGITS):
        result[i + 1] = temp[i]

if __name__ == "__main__":
    def multiply(a, b):
        num1 = to_chunks(a)
        num2 = to_chunks(b)
        result = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
        
        # Launch kernel with one block and one thread
        multiply_chunks_kernel[1, 1](num1, num2, result)
        return result

    a = '1.234567890123456789'
    b = '-1.234567890123456789'

    print(multiply(a, b))
    print(chunks_to_decimal(multiply(a, b)))
