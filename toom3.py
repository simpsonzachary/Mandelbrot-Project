import numpy as np
from numba import cuda

from utils import to_chunks, chunks_to_decimal, MAX_DIGITS, MAX_DIGITS_TEMP, TOOM3_CHUNK_DIGITS

@cuda.jit
def multiply_chunks_kernel(a, b, result):
    if (a[0] == b[0]):
        result[0] = 0
    else:
        result[0] = 1
        
    temp = cuda.local.array(shape=(MAX_DIGITS_TEMP), dtype=np.int16)
    
    a0 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    a1 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    a2 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    
    b0 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    b1 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    b2 = cuda.local.array(shape=(TOOM3_CHUNK_DIGITS), dtype=np.int16)
    
    for i in range(TOOM3_CHUNK_DIGITS):
        a0[i] = a[i + 1]
        a1[i] = a[i + TOOM3_CHUNK_DIGITS + 1]
        a2[i] = a[i + 2 * TOOM3_CHUNK_DIGITS + 1]
        b0[i] = b[i + 1]
        b1[i] = b[i + TOOM3_CHUNK_DIGITS + 1]
        b2[i] = b[i + 2 * TOOM3_CHUNK_DIGITS + 1]
    
    # a0 * b0
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(0, TOOM3_CHUNK_DIGITS):
            temp[i + j] += a0[i] * b0[j]
    # a0 * b1
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(0, TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 1] += a0[i] * b1[j]
    # a0 * b2
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(0, TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 2] += a0[i] * b2[j]
    # a1 * b0
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 1] += a1[i] * b0[j]
    # a1 * b1
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 2] += a1[i] * b1[j]
    # a1 * b2
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 3] += a1[i] * b2[j]
    # a2 * b0
    for i in range(0, TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 2] += a2[i] * b0[j]
    # a2 * b1
    for i in range(0, TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 3] += a2[i] * b1[j]
    # a2 * b2
    for i in range(TOOM3_CHUNK_DIGITS):
        for j in range(TOOM3_CHUNK_DIGITS):
            temp[i + j + TOOM3_CHUNK_DIGITS * 4] += a2[i] * b2[j]
            
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
        result = np.zeros(MAX_DIGITS + 1, dtype=np.int16)
        
        # Launch kernel with one block and one thread
        multiply_chunks_kernel[1, 1](num1, num2, result)
        return result

    a = '1.234567890123456789'
    b = '-1.234567890123456789'

    print(multiply(a, b))
    print(chunks_to_decimal(multiply(a, b)))
