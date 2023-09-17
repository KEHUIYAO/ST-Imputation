import numpy as np
import matplotlib.pyplot as plt

# bisquare function
def bisquare(x, c):
    # Calculate bisquare weight for each value x
    weight = np.where(
        np.abs(x) < c,
        (1 - (x / c) ** 2) ** 2,
        0
    )
    return weight

# Create an array of values (from -3 to 3)
x_values = np.linspace(-3, 3, 500)

# Choose a tuning constant c (e.g., 1)
c = 0.5

# Calculate the bisquare weights
weights = bisquare(x_values, c)

# Plot the bisquare function
plt.figure()
plt.plot(x_values, weights)
plt.title('Bisquare Function')
plt.xlabel('x')
plt.ylabel('w(x) (weight)')
plt.grid(True)
plt.show()

# cosine function


import numpy as np
import matplotlib.pyplot as plt

def bisquare_2d(x, y, c):
    norm = np.sqrt(x**2 + y**2)
    weight = np.where(
        norm < c,
        (1 - (norm / c) ** 2) ** 2,
        0
    )
    return weight

# Create a grid of x and y values
x_values = np.linspace(-3, 3, 500)
y_values = np.linspace(-3, 3, 500)
x, y = np.meshgrid(x_values, y_values)

# Choose a tuning constant c (e.g., 1)
c = 0.5

# Calculate the bisquare weights
weights = bisquare_2d(x, y, c)

# Plot the 2D bisquare function
plt.figure()
plt.contourf(x, y, weights, levels=50, cmap='viridis')
plt.colorbar(label='Weight')
plt.title('2D Bisquare Function')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generate an array of x values
x = np.linspace(0, 4*np.pi, 1000)  # from 0 to 4*pi

# Generate y values based on the cosine functions
y1 = np.cos(x)
y2 = np.cos(2*x)

# Generate y values for cos(nx) for n = 3, 4, 5
y3 = np.cos(3*x)
y4 = np.cos(4*x)
y5 = np.cos(0.2*x)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot cos(x)
plt.subplot(3, 2, 1)
plt.plot(x, y1)
plt.title('cos(x)')
plt.grid(True)

# Plot cos(2x)
plt.subplot(3, 2, 2)
plt.plot(x, y2)
plt.title('cos(2x)')
plt.grid(True)

# Plot cos(3x)
plt.subplot(3, 2, 3)
plt.plot(x, y3)
plt.title('cos(3x)')
plt.grid(True)

# Plot cos(4x)
plt.subplot(3, 2, 4)
plt.plot(x, y4)
plt.title('cos(4x)')
plt.grid(True)

# Plot cos(0.2x)
plt.subplot(3, 2, 5)
plt.plot(x, y5)
plt.title('cos(0.2x)')
plt.grid(True)

plt.tight_layout()
plt.show()
