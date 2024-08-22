from decimal import Decimal, getcontext
import numpy as np
from numba import cuda
import time
import math
from PIL import Image

from multiply import multiply_chunks_kernel
from add import add_chunks_kernel
from utils import to_chunks, chunks_to_decimal, MAX_DIGITS
        
@cuda.jit
def copy_array(a, b):
    for i in range(0, MAX_DIGITS):
        a[i] = b[i]
        
@cuda.jit 
def mandelbrot_kernel(imag_values, real_values, image, res, max_iterations, final_real, final_imag):
    x, y = cuda.grid(2)
                
    if x > res or y > res:
        return
                
    z_real = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    z_imag = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    z_real2 = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    z_imag2 = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    new_z_real = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)
    result = cuda.local.array(shape=(MAX_DIGITS), dtype=np.int32)

    negative1 = cuda.shared.array(shape=(MAX_DIGITS), dtype=np.int32)
    negative1[0] = 1
    negative1[1] = 1
                
    two = cuda.shared.array(shape=(MAX_DIGITS), dtype=np.int32)
    two[1] = 2
                
    for k in range(max_iterations):
                    
        multiply_chunks_kernel(z_real, z_real, z_real2)
        multiply_chunks_kernel(z_imag, z_imag, z_imag2)
                    
        add_chunks_kernel(z_real2, z_imag2, result)
                    
        if (result[1] >= 4):
            copy_array(final_real[x][y], z_real)
            copy_array(final_imag[x][y], z_imag)
            image[x][y] = k
            return
                    
        multiply_chunks_kernel(z_imag2, negative1, result)
        add_chunks_kernel(z_real2, result, result)
        add_chunks_kernel(result, real_values[x][y], new_z_real)
                    
        multiply_chunks_kernel(two, z_real, result)
        multiply_chunks_kernel(result, z_imag, result)
        add_chunks_kernel(result, imag_values[x][y], z_imag)
                    
        copy_array(z_real, new_z_real)
                    
    image[x][y] = max_iterations
    return
                
def generate_mandelbrot(center_x, center_y, zoom, res, max_iterations):
    start_time = time.time()
    image = np.zeros((res, res), dtype=np.int32)
    
    real_values = np.zeros((res, res, MAX_DIGITS + 1), dtype=np.int32)
    imag_values = np.zeros((res, res, MAX_DIGITS + 1), dtype=np.int32)
    
    final_real = np.zeros((res, res, MAX_DIGITS + 1), dtype=np.int32)
    final_imag = np.zeros((res, res, MAX_DIGITS + 1), dtype=np.int32)
    
    print("Setting pixels...")
    real_values, imag_values = set_pixels(center_x, center_y, real_values, imag_values, zoom, res)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to set pixels: {elapsed_time} seconds")
    
    d_image = cuda.to_device(image)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(res / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(res / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    start_time = time.time()
    print("Calculating grid...")
    mandelbrot_kernel[blockspergrid, threadsperblock](imag_values, real_values, d_image, res, max_iterations, final_real, final_imag)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to calculate grid: {elapsed_time} seconds")
    
    image = d_image.copy_to_host()
    return image, final_real, final_imag, real_values, imag_values

def set_pixels(center_x_str, center_y_str, real_values, imag_values, zoom_str, res):
    
    center_x = Decimal(center_x_str)
    center_y = Decimal(center_y_str) * -1
    zoom = Decimal(zoom_str)
    
    pixel_size = Decimal(3.0) * zoom / res
    minX = center_x - (pixel_size * res / Decimal(2.0))
    minY = center_y - (pixel_size * res / Decimal(2.0))

    real_values = [[0 for _ in range(res)] for _ in range(res)]
    imag_values = [[0 for _ in range(res)] for _ in range(res)]
    
    # Fill grids
    for i in range(res):
        for j in range(res):
            x = minX + (i * pixel_size)
            y = minY + (j * pixel_size)
            x = '{:.50f}'.format(x)
            y = '{:.50f}'.format(y)
            real_values[j][i] = to_chunks(str(x))
            imag_values[j][i] = to_chunks(str(y))

    real_values = np.array(real_values)
    imag_values = np.array(imag_values)

    return real_values, imag_values

def array_to_image(array, gradient, max_value, final_real, final_imag, real_values, imag_values):
    start_time = time.time()
    image_data = np.zeros((res, res, 3), dtype=np.uint8)

    for i in range(res):
        for j in range(res):
            if(array[i][j] == max_value):
                image_data[i, j, 0] = 0
                image_data[i, j, 1] = 0
                image_data[i, j, 2] = 0
            else:
                z_real = chunks_to_decimal(final_real[i][j])
                z_imag = chunks_to_decimal(final_imag[i][j])

                nsmooth = float(array[i][j] - (math.log(math.log(math.sqrt(z_real * z_real + z_imag * z_imag)))) / math.log(2))
                color_index = int(math.sqrt(nsmooth) * 96) % (gradient.shape[0])
                
                image_data[i, j, 0] = gradient[color_index, 0]
                image_data[i, j, 1] = gradient[color_index, 1]
                image_data[i, j, 2] = gradient[color_index, 2]
            
    # Create and save the image
    img = Image.fromarray(image_data, 'RGB')  # 'L' mode is for grayscale
    img.save('output_image.png')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to assign colors: {elapsed_time} seconds")

def generate_gradient(colors, num_steps):
    num_colors = len(colors)
    gradient = np.zeros((num_steps, 3), dtype=np.int16)
    
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
start_time = time.time()

center_x = '0'
center_y = '0'
zoom     = '1'

res = 240
max_iterations = 100

colors = [
    (0, 7, 100),
    (32, 107, 203), 
    (237, 255, 255), 
    (255, 170, 0), 
    (0, 2, 0),
]
num_steps = 2048
gradient = generate_gradient(colors, num_steps)

result, final_real, final_imag, real_values, imag_values = generate_mandelbrot(center_x, center_y, zoom, res, max_iterations)
    
print("Applying color...")
array_to_image(result, gradient, max_iterations, final_real, final_imag, real_values, imag_values)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time: {elapsed_time} seconds")