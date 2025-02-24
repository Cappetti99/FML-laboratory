import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Definiamo i punti di controllo
y = np.array([85,69, 60, 48,51, 46, 43,46,115])  # Punti x
x = np.array([1, 3, 5, 7, 9,11,13,15,17])  # Punti y

# Creiamo la spline cubica
cs = CubicSpline(x, y)

# Generiamo un set di punti più fitti per disegnare la curva
x_fine = np.linspace(1, 17)
y_fine = cs(x_fine)

# Visualizziamo la spline e i punti originali
plt.figure(figsize=(8, 6))
plt.plot(x_fine, y_fine, label='Spline Cubica', color='b')  # La spline
plt.scatter(x, y, color='r', label='Punti di controllo')   # I punti di controllo
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
