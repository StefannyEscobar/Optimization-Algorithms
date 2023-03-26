import numpy as np

# Función a minimizar
def f(x):
    return  100*((x[1] - x[0]**2)**2) +(1 - x[0])**2

# Gradiente de la función
def gradiente(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

# Hessiana de la función
def hessiana(x):
    return np.array([[1200*x[0]**2-2-400*x[1], -400*x[0]], [-400*x[0], 200]])

# Descenso de gradiente
def descenso_gradiente(gradiente, p_inicial, error, max_iter):
    x = p_inicial
    iteracion = 0
    while iteracion < max_iter:
        g = gradiente(x)
        lam = np.dot(g.T,g)/(2*np.dot(np.dot(g.T, hessiana(x)), g))
        x = x - lam*g
        norma_g = np.linalg.norm(g)
        print(f"Iteración {iteracion+1} | x={x} | f(x)={f(x)}|  norma del gradiente={norma_g}")
        if norma_g < error:
            break
        iteracion += 1
    return x

# Ejemplo de uso
p_inicial = np.array([-1.2, 1])
x_min = descenso_gradiente(gradiente, p_inicial, error=1e-4, max_iter=10000)
print("\nMínimo encontrado en:", x_min)
print("Valor mínimo de la función:", f(x_min))
