import numpy as np
from numba import cuda
from PIL import Image
import cv2
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
    height, width = image.shape[:2]
    startX, startY = cuda.grid(2)
    
    # Calculate pixel size
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    z_real = 0
    z_imag = 0
    
    for x in range(startX, width, cuda.blockDim.x * cuda.gridDim.x):
        c_real = min_x + x * pixel_size_x
        for y in range(startY, height, cuda.blockDim.y * cuda.gridDim.y):
            c_imag = min_y + y * pixel_size_y
            curr_iteration = 0
            
            for k in range(max_iter):
                check = z_real * z_real + z_imag * z_imag
                if check >= 4:
                    break
                z_real_sq = z_real * z_real - z_imag * z_imag
                z_imag = 2 * z_real * z_imag + c_imag
                z_real = z_real_sq + c_real
                
                curr_iteration += 1
    
            if curr_iteration == max_iter:
                image[y, x, 0] = 0
                image[y, x, 1] = 0
                image[y, x, 2] = 0
            else:
                gradient_index = curr_iteration % gradient.shape[0]
                image[y, x, 0] = gradient[gradient_index, 0]
                image[y, x, 1] = gradient[gradient_index, 1]
                image[y, x, 2] = gradient[gradient_index, 2]

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
    range_x = max_x - min_x
    range_y = max_y - min_y

    new_range_x = range_x * zoom_level
    new_range_y = range_y * zoom_level

    min_x_new = center_x - new_range_x / 2
    max_x_new = center_x + new_range_x / 2
    min_y_new = center_y - new_range_y / 2
    max_y_new = center_y + new_range_y / 2

    return min_x_new, max_x_new, min_y_new, max_y_new


# Main portion
width, height = 1920, 1080
max_iter = 100

center_x = -0.5
center_y =  0
zoom_level = 1

colors = [
    (0, 2, 0),
    (0, 7, 100),
    (32, 107, 203), 
    (237, 255, 255), 
    (255, 170, 0), 
]

num_steps = 12

gradient = generate_gradient(colors, num_steps)

image = generate_mandelbrot(center_x, center_y, zoom_level, width, height, max_iter, gradient)

# Save the image as a PNG file
cv2.imshow('Generated Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()