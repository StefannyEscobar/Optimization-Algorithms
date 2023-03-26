import numpy as np

def armijo_rule(x, f, grad_f, t=1, alpha=0.5, beta=0.5):
    #Aplicamos el algorítmo de armijo. Condición:
    #f(xk + tdk) ≤ f(xk) + σ1t∇f(xk)
    while (f(x-t*grad_f(x))) > (f(x)-alpha*t*np.linalg.norm(grad_f(x))**2):
        t = beta*t
    return t

def newton_direction(x, hessian_f, grad_f):
    # Dirección de descenso de Newton
    #-Hess[f(x)]*∇f(x)
    p = -np.linalg.solve(hessian_f(x), grad_f(x))
    return p

def newton_armijo(x0, f, grad_f, hessian_f, tol=1e-8, max_iter=1000, t=1, alpha=0.5, beta=0.5):
    x = x0.copy()
    iter = 0
    while iter < max_iter:
        # Calcula dirección de descenso con método de Newton
        p = newton_direction(x, hessian_f, grad_f)
        # Calcula el tamaño de paso con la regla de Armijo
        t = armijo_rule(x, f, grad_f, t=t, alpha=alpha, beta=beta)
        # Actualización del punto
        #xn+1 = xn+t * -Hess[f(x)]*∇f(x)
        x = x + t*p
        # Verifica el criterio de parada
        if np.linalg.norm(grad_f(x)) < tol:
            break
        iter += 1
    return x, f(x)


# Función objetivo
def f(x):
    return 5*x[0]**2 + 5*x[1]**2 - x[0]*x[1] - 11*x[0] + 11*x[1] + 11

#Gradiente de la función
def grad_f(x):
    grad = np.zeros(2)
    grad[0] = 10*x[0] - x[1] - 11
    grad[1] = 10*x[1] - x[0] + 11
    return grad

#Hessiano de la función
def hess_f(x):
    hess = np.zeros((2,2))
    hess[0,0] = 10
    hess[0,1] = -1
    hess[1,0] = -1
    hess[1,1] = 10
    return hess

# Punto inicial
x0 = np.array([0, 0])

x_opt, f_opt = newton_armijo(x0, f, grad_f, hess_f)
print("Puntos: ", x_opt)
print("Valor óptimo: ", f_opt)
