from decimal import Decimal, getcontext
import numpy as np
from numba import cuda
import math
from PIL import Image

MAX_DIGITS = 1000

@cuda.jit
def add_chunks_kernel(a, b, result, temp, digits):
        if(a[0] == b[0]):
            temp[0] = a[0]
            for i in range(digits, 0, -1):
                temp[i] += a[i] + b[i]
                if(temp[i] >= 10):
                    temp[i - 1] += 1
                    temp[i] -= 10
                    
        greater_abs = 0
        for i in range(1, digits):
            if (a[i] > b[i]):
                break
            if (b[i] > a[i]):
                greater_abs = 1
                break
        
        if (a[0] == 1 and b[0] == 0):
            if (greater_abs == 0):
                temp[0] = 1
                for i in range(digits, 0, -1):
                    temp[i] += a[i] - b[i]
                    if(temp[i] < 0):
                        temp[i - 1] -= 1
                        temp[i] += 10
            if (greater_abs == 1):
                temp[0] = 0
                for i in range(digits, 0, -1):
                    temp[i] += b[i] - a[i]
                    if(temp[i] < 0):
                        temp[i - 1] -= 1
                        temp[i] += 10
                        
        if (a[0] == 0 and b[0] == 1):
            if (greater_abs == 0):
                temp[0] = 0
                for i in range(digits, 0, -1):
                    temp[i] += a[i] - b[i]
                    if(temp[i] < 0):
                        temp[i - 1] -= 1
                        temp[i] += 10
            if (greater_abs == 1):
                temp[0] = 1
                for i in range(digits, 0, -1):
                    temp[i] += b[i] - a[i]
                    if(temp[i] < 0):
                        temp[i - 1] -= 1
                        temp[i] += 10
        
        for i in range(0, digits):
            result[i] = temp[i]
        
        for i in range(0, (digits * 2) + 1):
            temp[i] = 0

@cuda.jit
def multiply_chunks_kernel(a, b, result, temp, digits):
        if (a[0] == b[0]):
            result[0] = 0
        else:
            result[0] = 1
        
        for i in range(0, digits):
            for j in range(0, digits):
                temp[i + j] += a[i + 1] * b[j + 1]
        
        for i in range(digits * 2, 0, -1):
            if(temp[i] >= 10):
                temp[i - 1] += temp[i] / 10
                temp[i] %= 10
                
        for i in range(0, digits):
            result[i + 1] = temp[i]
        
        for i in range(0, (digits * 2) + 1):
            temp[i] = 0

@cuda.jit
def reset_array(arr, digits):
    for i in range(0, digits):
        arr[i] = 0
        
@cuda.jit
def copy_array(a, b, digits):
    for i in range(0, digits):
        a[i] = b[i]
@cuda.jit 
def mandelbrot_single_point(imag_values, real_values, image, res, digits, max_iterations):
                x, y = cuda.grid(2)
                
                if x >= res or y >= res:
                    return
                
                z_real = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                z_imag = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                z_real2 = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                z_imag2 = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                new_z_real = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                new_z_imag = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                result = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                temp = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)

                negative1 = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                negative1[0] = 1
                negative1[1] = 1
                
                two = cuda.local.array(shape=(MAX_DIGITS,), dtype=np.int32)
                two[1] = 2
                
                reset_array(z_real, digits)
                reset_array(z_imag, digits)
                for k in range(max_iterations + 1):
                    
                    if(k == max_iterations):
                        image[x][y] = k
                        break
                    
                    multiply_chunks_kernel(z_real, z_real, z_real2, temp, digits)
                    multiply_chunks_kernel(z_imag, z_imag, z_imag2, temp, digits)
                    
                    add_chunks_kernel(z_real2, z_imag2, result, temp, digits)
                    
                    magnitude_squared = result[1]
                    if (magnitude_squared >= 4):
                        image[x][y] = k
                        break
                    reset_array(result, digits)
                    multiply_chunks_kernel(z_imag2, negative1, result, temp, digits)
                    add_chunks_kernel(z_real2, result, result, temp, digits)
                    add_chunks_kernel(result, real_values[x][y], new_z_real, temp, digits)
                    reset_array(result, digits)
                    
                    multiply_chunks_kernel(two, z_real, result, temp, digits)
                    multiply_chunks_kernel(result, z_imag, result, temp, digits)
                    add_chunks_kernel(result, imag_values[x][y], new_z_imag, temp, digits)
                    
                    copy_array(z_real, new_z_real, digits)
                    copy_array(z_imag, new_z_imag, digits)
                    
                    # Reset result for next iteration
                    reset_array(result, digits)
                    reset_array(z_real2, digits)
                    reset_array(z_imag2, digits)
                    reset_array(new_z_real, digits)
                    reset_array(new_z_imag, digits)
                
                
def generate_mandelbrot(center_x, center_y, zoom, res, max_iterations, digits):
    image = np.zeros((res, res), dtype=np.int32)
    real_values = np.zeros((res, res, digits + 1))
    imag_values = np.zeros((res, res, digits + 1))
    
    real_values, imag_values = set_pixels(center_x, center_y, real_values, imag_values, zoom, res, digits)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(res / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(res / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    mandelbrot_single_point[blockspergrid, threadsperblock](imag_values, real_values, image, res, digits, max_iterations)
    return image

def set_pixels(center_x_str, center_y_str, real_values, imag_values, zoom_str, res, digits):

    getcontext().prec = digits * 2
    
    # Convert input strings to Decimal
    center_x = Decimal(center_x_str)
    center_y = Decimal(center_y_str) * -1
    zoom = Decimal(zoom_str)
    
    # Calculate pixel size and minimum coordinates
    pixel_size = Decimal(3.0) * zoom / Decimal(res)
    minX = center_x - (pixel_size * Decimal(res) / Decimal(2))
    minY = center_y - (pixel_size * Decimal(res) / Decimal(2))
    
    # Initialize grids
    real_values = [[None for _ in range(res)] for _ in range(res)]
    imag_values = [[None for _ in range(res)] for _ in range(res)]
    
    # Fill grids
    for i in range(res):
        for j in range(res):
            x = minX + (Decimal(i) * pixel_size)
            y = minY + (Decimal(j) * pixel_size)
            x = '{:.50f}'.format(x)
            y = '{:.50f}'.format(y)
            real_values[j][i] = to_chunks(str(x), digits)
            imag_values[j][i] = to_chunks(str(y), digits)

    real_values = np.array(real_values)
    imag_values = np.array(imag_values)

    return real_values, imag_values
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

def array_to_image(array, gradient, max_value):
    height, width = array.shape
    image_data = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            image_data[i, j, 0] = gradient[array[i][j] % 2048, 0]
            image_data[i, j, 1] = gradient[array[i][j] % 2048, 1]
            image_data[i, j, 2] = gradient[array[i][j] % 2048, 2]
            if(array[i][j] == max_value):
                image_data[i, j, 0] = 0
                image_data[i, j, 1] = 0
                image_data[i, j, 2] = 0
    # Create and save the image
    img = Image.fromarray(image_data, 'RGB')  # 'L' mode is for grayscale
    img.save('output_image.png')

def generate_gradient(colors, num_steps):
    num_colors = len(colors)
    gradient = np.zeros((num_steps, 3), dtype=np.uint8)
    
    # Calculate steps per segment
    steps_per_segment = num_steps / num_colors
    
    for i in range(num_colors):
        start_color = colors[i]
        end_color = colors[(i + 1) % num_colors]  # Loop back to the first color
        
        for j in range(int(steps_per_segment)):
            t = j / (steps_per_segment - 1)
            gradient_index = int(i * steps_per_segment + j)
            r = int(start_color[0] * (1 - t) + end_color[0] * t)
            g = int(start_color[1] * (1 - t) + end_color[1] * t)
            b = int(start_color[2] * (1 - t) + end_color[2] * t)
            
            gradient[gradient_index] = [r, g, b]
    
    return gradient

# main operating portion

num_digits = 24
center_x = '-0.7445398603559083806'
center_y = ' 0.1217237738944248242'
zoom     = '7.5e-16'

res = 1080
max_iterations = 50000

colors = [
    (255, 0, 0), #Red
    (0, 0, 0), #Black
    (255, 215, 0),   #Gold
]
num_steps = 2048
gradient = generate_gradient(colors, num_steps)

result = generate_mandelbrot(center_x, center_y, zoom, res, max_iterations, num_digits)

print(result)
array_to_image(result, gradient, max_iterations)