import numpy as np
from numba import cuda
from PIL import Image
import math

class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def to_tuple(self):
        return (self.r, self.g, self.b)
    
    def __repr__(self):
        return f"Color(r={self.r}, g={self.g}, b={self.b})"

@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, image, max_iter, gradient):
    height, width, _ = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    startX, startY = cuda.grid(2)
    
    for x in range(startX, width, cuda.blockDim.x * cuda.gridDim.x):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, cuda.blockDim.y * cuda.gridDim.y):
            imag = min_y + y * pixel_size_y
            c_real = np.float64(real)
            c_imag = np.float64(imag)
            z_real = np.float64(0.0)
            z_imag = np.float64(0.0)
            curr_iteration = 0
            while(z_real * z_real + z_imag * z_imag < 2 and curr_iteration < max_iter):
                z_real_sq = z_real * z_real - z_imag * z_imag
                z_imag = 2 * z_real * z_imag + c_imag
                z_real = z_real_sq + c_real
                curr_iteration += 1
         
            # Apply a color gradient based on the number of iterations
            if curr_iteration == max_iter:
                image[y, x, 0] = 0
                image[y, x, 1] = 0
                image[y, x, 2] = 0
            else:
                nsmooth = float(math.log(math.log((z_real * z_real) + (z_imag * z_imag))) / math.log(2))
                color_index = int(math.sqrt(curr_iteration - nsmooth) * 256) % gradient.shape[0]
                image[y, x, 0] = gradient[color_index, 0]
                image[y, x, 1] = gradient[color_index, 1]
                image[y, x, 2] = gradient[color_index, 2]

def generate_mandelbrot(center_x, center_y, zoom_level, width, height, max_iter, gradient):
    min_x, max_x = -2.0, 1.0
    min_y, max_y = -1.5, 1.5
    min_x, max_x, min_y, max_y = update_bounds(min_x, max_x, min_y, max_y, center_x, center_y, zoom_level)
    aspect_ratio = width / height
    if aspect_ratio > 1:
        range_x = max_x - min_x
        range_y = range_x / aspect_ratio
        center_y = (min_y + max_y) / 2
        min_y = center_y - range_y / 2
        max_y = center_y + range_y / 2
    else:
        range_y = max_y - min_y
        range_x = range_y * aspect_ratio
        center_x = (min_x + max_x) / 2
        min_x = center_x - range_x / 2
        max_x = center_x + range_x / 2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, image, max_iter, gradient)
    cuda.synchronize()
    
    return image

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

def update_bounds(min_x, max_x, min_y, max_y, center_x, center_y, zoom_level):
    # Calculate the range of the current view
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Update ranges based on the zoom level
    new_range_x = range_x * zoom_level
    new_range_y = range_y * zoom_level

    # Calculate new bounds centered around (center_x, center_y)
    min_x_new = center_x - new_range_x / 2
    max_x_new = center_x + new_range_x / 2
    min_y_new = center_y - new_range_y / 2
    max_y_new = center_y + new_range_y / 2

    return min_x_new, max_x_new, min_y_new, max_y_new

# Main portion
width, height = 2560, 1440
max_iter = 50000

center_x = -0.7445398603559061
center_y = 0.1217237738944253
zoom_level = 9.375e-16

colors = [
    (255, 255, 255), #White
    (0, 0, 0),   #Black
]
num_steps = 2048

gradient = generate_gradient(colors, num_steps)

image = generate_mandelbrot(center_x, center_y, zoom_level, width, height, max_iter, gradient)

# Save the image as a PNG file
im = Image.fromarray(image)
im.save('mandelbrot_colored.png')