from decimal import Decimal, getcontext
import numpy as np
from numba import cuda
import time
import math
from PIL import Image

from toom3 import multiply_chunks_kernel
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
    
    d_real = cuda.to_device(real_values)
    d_imag = cuda.to_device(imag_values)
    d_freal = cuda.to_device(final_real)
    d_fimag = cuda.to_device(final_imag)
    d_image = cuda.to_device(image)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(res / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(res / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    start_time = time.time()
    print("Calculating grid...")
    
    mandelbrot_kernel[blockspergrid, threadsperblock](d_imag, d_real, d_image, res, max_iterations, d_freal, d_fimag)
    
    real_values = d_real.copy_to_host()
    imag_values = d_imag.copy_to_host()
    final_real = d_freal.copy_to_host()
    final_imag = d_fimag.copy_to_host()
    image = d_image.copy_to_host()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to calculate grid: {elapsed_time} seconds")
    
    return image, final_real, final_imag, real_values, imag_values

def set_pixels(center_x_str, center_y_str, real_values, imag_values, zoom_str, res):
    
    getcontext().prec = MAX_DIGITS * 2
    
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
            x = '{:.200f}'.format(x)
            y = '{:.200f}'.format(y)
            real_values[j][i] = to_chunks(str(x))
            imag_values[j][i] = to_chunks(str(y))

    real_values = np.array(real_values)
    imag_values = np.array(imag_values)

    return real_values, imag_values

def array_to_image(array, gradient, max_value, final_real, final_imag):
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
                color_index = int(nsmooth * 6) % (gradient.shape[0])
                
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
            
    gradient[num_steps - 1] = colors[0]
    
    return gradient

# main operating portion
start_time = time.time()

center_x = '-0.154336052508107179265305091632196962650884871466568125910282277206471652431628774014739313859902112007245675469882225230637912621473564145409973045901'
center_y = '1.0307951891020566491064253562797191540856566236276766471523470911761717308962353343318675114228208177256898278834543858465101418159849468885389383082372'
zoom     = '4e-96'

res = 256
max_iterations = 5000

colors = [
    (100, 7, 0),
    (203, 107, 32), 
    (255, 255, 237), 
    (0, 170, 255), 
    (0, 2, 0),
]

num_steps = 256
gradient = generate_gradient(colors, num_steps)

result, final_real, final_imag, real_values, imag_values = generate_mandelbrot(center_x, center_y, zoom, res, max_iterations)
    
print("Applying color...")
array_to_image(result, gradient, max_iterations, final_real, final_imag)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time: {elapsed_time} seconds")