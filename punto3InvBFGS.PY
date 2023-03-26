import numpy as np

def f(x,mu):
    error = -1e-4
    y = x[0] +x[1] + 2*x[2] -3
    if y < error:
        return (9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 +
            2*x[0]*x[1] + 2*x[0]*x[2] -mu*1/(x[0] +x[1] + 2*x[2] -3)-mu/x[0]-mu/x[1]-mu/x[2])
    else:
            return (9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 +
            2*x[0]*x[1] + 2*x[0]*x[2] -4*mu*1/(error))
def fun(x):
    return 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2]


def grad(x,mu):
    error = -1e-4
    y = x[0]+x[1]+2*x[2]-3
    if y < error:
        return np.array([ -8+4*x[0]+2*x[1]+2*x[2] +mu*1/(x[0]+x[1]+2*x[2]-3)**2+mu/x[0]**2,
                    -6+4*x[1]+2*x[0] +mu*1/(x[0]+x[1]+2*x[2]-3)**2+mu/x[1]**2 ,
                    -4+2*x[2]+2*x[0] +2*mu*1/(x[0]+x[1]+2*x[2]-3)**2+mu/x[2]**2])
    else:
        return np.array([ -8+4*x[0]+2*x[1]+2*x[2],
            -6+4*x[1]+2*x[0],
            -4+2*x[2]+2*x[0]])

def wolfe_conditions(f,x,p,ngrad):
    # Backtrack line search con la condiciones Wolfe
    a = 0.9
    c1 = 1e-8
    c2 = 0.9
    eta = 1e-4
    mu = eta
    fx = f(x,mu)
    x_new = x + a * p
    ngrad_new = grad(x_new, mu)
    while f(x_new, mu) >= fx + (c1*a*ngrad.T@p) or ngrad_new.T@p <= c2*ngrad.T@p :
        a *= 0.50001
        x_new = x + a * p
        ngrad_new = grad(x_new, mu)
    return a

def BFGS(f,x0,max_it,fun):
    d = len(x0) # dimension del problema
    eta = 1e-4
    mu = eta
    ngrad = grad(x0,mu) # gradiente inicial
    H = np.eye(d) # aproximación hessiana incial
    x = x0[:]
    it = 0
    while np.linalg.norm(ngrad) > 1e-4: # condicion de parada
        if it >= max_it:
            print('Iteración máxima alcanzada')
            break
        it += 1
        p = -H@ngrad # Dirección por Newton
        a = wolfe_conditions(f,x,p,ngrad) # line search
        s = a * p
        x_new = x + a * p
        ngrad_new = grad(x_new, mu)
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
        dif = abs(x-x_new)
        print(dif)
        ngrad = ngrad_new[:]
        x = x_new[:]
        mu *= eta  # actualizar el valor de mu
        print("n:", it, " | P(x) =", f(x, mu), " | f(x) \u2248", fun(x), "| Mu: ",mu, "| x: ", x, "Mu*g(x)",-mu*1/(x[0] +x[1] + 2*x[2] -3)-mu/x[0]-mu/x[1]-mu/x[2] )
    return x
# resultado
x_opt = BFGS(f,[0.5,0.5,0.5],8,fun)
if (x_opt[0] +x_opt[1] + 2*x_opt[2])<3:
    print(f'los valores óptimos son:{x_opt} y f(x) \u2248 {fun(x_opt)}')
else:
    print(f'los valores {x_opt} no cumplen las condiciones pues {x_opt[0] +x_opt[1] + 2*x_opt[2]} !\u2264 3')