import matplotlib.pyplot as plt
import numpy as np

# Set style for dark theme
plt.style.use('dark_background')

# Create data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'c-', linewidth=2, label='sin(x)')
plt.fill_between(x, y, alpha=0.2, color='cyan')

# Customize plot
plt.title('Sine Wave Visualization', color='white', size=14)
plt.xlabel('x', color='white', size=12)
plt.ylabel('sin(x)', color='white', size=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Set background color
plt.gca().set_facecolor('#1e1e1e')
plt.gcf().set_facecolor('#1e1e1e')

# Adjust layout
plt.tight_layout()
