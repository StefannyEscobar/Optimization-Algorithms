import numpy as np

def f(x):
    # Función de prueba (rosenbrock)
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
def grad(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

def wolfe_conditions(f,x,p,ngrad):
    # Backtrack line search con la condiciones Wolfe
    a = 1
    c1 = 1e-7
    c2 = 0.92
    fx = f(x)
    x_new = x + a * p
    ngrad_new = grad(x_new)
    while f(x_new) >= fx + (c1*a*ngrad.T@p) or ngrad_new.T@p <= c2*ngrad.T@p :
        a *= 0.67
        x_new = x + a * p
        ngrad_new = grad(x_new)
    return a

def BFGS(f,x0,max_it):
    d = len(x0) # dimension del problema
    ngrad = grad(x0) # gradiente inicial
    H = np.eye(d) # aproximación hessiana incial
    x = x0[:]
    it = 0
    while np.linalg.norm(ngrad) > 1e-5: # condicion de parada
        if it > max_it:
            print('Iteración máxima alcanzada')
            break
        it += 1
        p = -H@ngrad # Dirección por Newton
        a = wolfe_conditions(f,x,p,ngrad) # line search
        s = a * p
        x_new = x + a * p
        ngrad_new = grad(x_new)
        y = ngrad_new - ngrad
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS actualizado
        ngrad = ngrad_new[:]
        x = x_new[:]
        print("Iteración:", it, " | f(x) \u2248", f(x), " | norma del gradiente =", np.linalg.norm(ngrad))
    return x
# resultado
x_opt = BFGS(f,[-1.2,1],100)
print(f'los valores óptimos son:{x_opt} y f(x)\u2248 {f(x_opt)}')
