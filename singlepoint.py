from decimal import Decimal, getcontext

def mandelbrot_escape_iterations(x, y, max_iterations, precision=50):
    # Set precision for the decimal calculations
    getcontext().prec = precision
    
    # Initialize the complex number c = x + yi
    c_real = Decimal(x)
    c_imag = Decimal(y)
    
    # Initialize z = 0 + 0i
    z_real = Decimal(0)
    z_imag = Decimal(0)
    
    # Iterate to determine escape count
    for i in range(max_iterations):
        # Calculate z^2 = (z_real + z_imag * i)^2
        z_real_squared = z_real * z_real
        z_imag_squared = z_imag * z_imag
        z_real_new = z_real_squared - z_imag_squared + c_real
        z_imag_new = Decimal(2) * z_real * z_imag + c_imag
        
        # Update z to the new value
        z_real = z_real_new
        z_imag = z_imag_new
        if(i == 2): 
            print({z_real})
        # Check if the magnitude of z exceeds 2 (escape condition)
        if z_real_squared + z_imag_squared > Decimal(4):
            return i
    
    # If the loop completes, the point is likely in the Mandelbrot set
    return max_iterations

# Example usage:
x = Decimal('-0.7445398603559083829')  # Example x value
y = Decimal('0.1217237738944248242')  # Example y value
max_iterations = 50000  # Set the maximum number of iterations

iterations = mandelbrot_escape_iterations(x, y, max_iterations)
print(f"Escape iterations for point ({x}, {y}): {iterations}")
