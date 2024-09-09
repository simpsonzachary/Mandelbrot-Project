import math

# Constants
theta_mas = 3  # angular diameter in milli arcseconds
distance_ly = 16.7  # distance in light years
ly_to_km = 9.461e12  # 1 light year in meters

# Convert angular diameter from milli arcseconds to radians
theta_radians = theta_mas * (math.pi / (180 * 3600 * 1000))

# Convert distance from light years to meters
distance_km = distance_ly * ly_to_km

# Calculate actual diameter D using the formula D = 2 * d * tan(θ / 2)
actual_diameter = 2 * distance_km * math.tan(theta_radians / 2)

print(actual_diameter)