import numpy as np
from numba import cuda

from utils import to_chunks, chunks_to_decimal, MAX_DIGITS

@cuda.jit
def add_chunks_kernel(a, b, result):
    temp = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    if(a[0] == b[0]):
        temp[0] = a[0]
        for i in range(MAX_DIGITS, 0, -1):
            temp[i] += a[i] + b[i]
            if(temp[i] >= 10 and i > 1):
                temp[i - 1] += 1
                temp[i] -= 10
                    
    greater_abs = 0
    for i in range(1, MAX_DIGITS):
        if (a[i] > b[i]):
            break
        if (b[i] > a[i]):
            greater_abs = 1
            break
        
    if (a[0] == 1 and b[0] == 0):
        if (greater_abs == 0):
            temp[0] = 1
            for i in range(MAX_DIGITS, 0, -1):
                temp[i] += a[i] - b[i]
                if(temp[i] < 0 and i > 1):
                    temp[i - 1] -= 1
                    temp[i] += 10
        if (greater_abs == 1):
            temp[0] = 0
            for i in range(MAX_DIGITS, 0, -1):
                temp[i] += b[i] - a[i]
                if(temp[i] < 0 and i > 1):
                    temp[i - 1] -= 1
                    temp[i] += 10
                        
    if (a[0] == 0 and b[0] == 1):
        if (greater_abs == 0):
            temp[0] = 0
            for i in range(MAX_DIGITS, 0, -1):
                temp[i] += a[i] - b[i]
                if(temp[i] < 0):
                    temp[i - 1] -= 1
                    temp[i] += 10
        if (greater_abs == 1):
            temp[0] = 1
            for i in range(MAX_DIGITS, 0, -1):
                temp[i] += b[i] - a[i]
                if(temp[i] < 0):
                    temp[i - 1] -= 1
                    temp[i] += 10
        
    for i in range(0, MAX_DIGITS):
        result[i] = temp[i]

if __name__ == "__main__":
    def add(a, b):
        num1 = to_chunks(a)
        num2 = to_chunks(b)
        result = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
        
        # Launch kernel with one block and one thread
        add_chunks_kernel[1, 1](num1, num2, result)
        return result

    a = '-4.12345678'
    b = '4.12345678'

    print(add(a, b))
    print(chunks_to_decimal(add(a, b)))
