import numpy as np
from numba import cuda

@cuda.jit
def add_chunks_kernel(a, b, result, digits):
    idx = cuda.grid(1)
    
    # Ensure we only use thread 0 to handle the entire computation
    if idx == 0:
        # Determine signs of numbers
        sign_a = a[0]  # 0 for positive, 1 for negative
        sign_b = b[0]  # 0 for positive, 1 for negative

        # Extract digits ignoring the sign
        a_digits = a[1:]
        b_digits = b[1:]

        if sign_a == sign_b:
            # Both numbers have the same sign, perform addition
            carry = 0
            for i in range(digits - 1, -1, -1):
                temp_sum = a_digits[i] + b_digits[i] + carry
                if temp_sum >= 10:
                    carry = temp_sum // 10
                    result[i] = temp_sum % 10
                else:
                    carry = 0
                    result[i] = temp_sum

            if carry > 0 and digits > 0:
                result[0] += carry
                if result[0] >= 10:
                    carry = result[0] // 10
                    result[0] %= 10
                    for i in range(1, digits):
                        result[i] += carry
                        if result[i] < 10:
                            carry = 0
                            break
                        carry = result[i] // 10
                        result[i] %= 10

            # Adjust sign based on the original signs
            if sign_a == 1:  # Both were negative
                result[0] = 1  # Result should be negative
        else:
            # Handle case where signs are different, perform subtraction
            def subtract_digits(larger, smaller, digits):
                carry = 0
                for i in range(digits - 1, -1, -1):
                    temp_diff = larger[i] - smaller[i] - carry
                    if temp_diff < 0:
                        carry = 1
                        result[i] = temp_diff + 10
                    else:
                        carry = 0
                        result[i] = temp_diff

            def compare_digit_arrays(a_digits, b_digits, digits):
                for i in range(digits):
                    if a_digits[i] > b_digits[i]:
                        return True
                    elif a_digits[i] < b_digits[i]:
                        return False
                return False

            if compare_digit_arrays(a_digits, b_digits, digits):
                larger = a_digits
                smaller = b_digits
                result_sign = sign_a
            else:
                larger = b_digits
                smaller = a_digits
                result_sign = sign_b

            subtract_digits(larger, smaller, digits)

            # Remove leading zeros in result if necessary
            first_nonzero = 0
            for i in range(digits):
                if result[i] != 0:
                    first_nonzero = i
                    break
            result[:] = result[first_nonzero:]  # Remove leading zeros
            if len(result) == 0:  # Handle zero case
                result[:] = np.array([0], dtype=np.int32)
            result[0] = result_sign

def add(num1, num2, digits):
    result = np.zeros(digits + 1, dtype=np.int32)
    threads_per_block = 1  # Only one thread
    blocks_per_grid = 1  # Only one block
    
    d_num1 = cuda.to_device(num1)
    d_num2 = cuda.to_device(num2)
    d_result = cuda.device_array(digits + 1, dtype=np.int32)

    add_chunks_kernel[blocks_per_grid, threads_per_block](d_num1, d_num2, d_result, digits)
    
    result = d_result.copy_to_host()
    return result


# Example usage
num_digits = 24  # Example number of digits
a = np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # Positive number
b = np.array([1, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # Negative number

result = add(a, b, num_digits)
print("Result:", result)
