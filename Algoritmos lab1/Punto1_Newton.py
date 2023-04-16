import numpy as np

def f(x):
    return  100*((x[1] - x[0]**2)**2) +(1 - x[0])**2

def grad_f(x, y):
    return np.array([-400*x*(y - x**2) - 2*(1 - x), 200*(y - x**2)])

def hess_f(x, y):
    return np.array([[1200*x**2 - 400*y + 2, -400*x], [-400*x, 200]])

def newton(x0, max_iter,xopt):
    x = x0[:]
    for i in range(max_iter):
        d = np.linalg.solve(hess_f(x[0], x[1]), grad_f(x[0], x[1]))
        x_new = x - d
        norma_error = np.linalg.norm(x-xopt)
        print(f"Iteración {i+1}: x = {x_new} | f(x) = {f(x_new)} | Error = {norma_error}")
        if np.linalg.norm(x_new - xopt) < 1e-6:
            break
        x = x_new[:]

    return x_new,norma_error

x_opt,norma_error = newton(np.array([-1.2, 1]), 100,np.array([1, 1]))
print(f"\nSolución: x = {x_opt} | f(x) = {f(x_opt)} | Error = {norma_error}")