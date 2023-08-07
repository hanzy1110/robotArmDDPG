import jax
import jax.numpy as jnp
import numpy as np

def Tq(thetas, ls):
    tq = jnp.array([ls[0]*jnp.cos(thetas[0]) + ls[1]*jnp.cos(thetas[:-1].sum()) + ls[2]*jnp.cos(thetas.sum()),
                   ls[0]*jnp.sin(thetas[0]) + ls[1]*jnp.sin(thetas[:-1].sum()) + ls[2]*jnp.sin(thetas.sum())])
    return tq

def stabilize(J, eps=.1):
    return J+jnp.eye(J.shape[0])*eps

def newtonIter(q, J, dT, alpha=0.1):
    #Pseudo Inverse
    Jplus = np.linalg.inv(stabilize(J.T@J))@J.T
    return (q-Jplus.dot(dT)*alpha)%(2*jnp.pi)

def solve(goal, theta_init, niters, tol,ls):
    jacobian = jax.jacrev(Tq)
    # theta_init = jnp.zeros(3)

    i = 0
    theta = theta_init
    # theta = jnp.zeros(3)
    while i<niters:
        tApprox = Tq(theta, ls)
        J = jacobian(theta, ls)
        dT = goal-tApprox

        theta_new = newtonIter(theta, J, dT)

        # diff = theta-theta_new
        # diff2 = (eps:=jnp.linalg.norm(diff)/jnp.linalg.norm(theta_new))
        if (eps:=jnp.linalg.norm(dT)/jnp.linalg.norm(goal))<tol:
            params = {'error':eps,
                   'dT':dT, 
                   'tApprox':tApprox, 
                   'theta':theta, 
                   'theta_new':theta_new}

            print(params)
            print("Done!")
            return theta_new, tApprox
        else:
            # print('iter:',i)
            theta = theta_new
        i+=1

    params = {'iter':i,'error':eps,
                   'dT':dT, 
                   'tApprox':tApprox, 
                   'theta':theta, 
                   'theta_new':theta_new}

    print(params)

    print("Not converged!")
    return theta, tApprox

class NewtonControl:
    def __init__(self, ls, center_coord):
        self.ls=ls['l']
        self.center_coord = center_coord

    def correctGoal(self, goal):
        return jnp.array([goal['x']-self.center_coord[0], goal['y']-self.center_coord[1]])

    def get_angles(self, goal, actual):
        theta, tApprox = solve(goal,actual, 100, 1e-3, ls=self.ls)
        return theta, tApprox

    def get_action(self,goal, actual, alpha=0.1):
        goal = self.correctGoal(goal)
        angles, tApprox = self.get_angles(goal, actual)
        print(f"Goal {goal}")
        print(f"ApproxPoint = {tApprox}, Solution = {angles} ")
        action = actual-angles
        return actual + alpha*action

if __name__=="__main__":
    theta_init = jnp.array([np.pi/3, -np.pi/6, -np.pi/6])
    ls = np.ones(3)*100
    objective=Tq(theta_init, ls)*0.9
    theta, tApprox = solve(objective, 1000, 1e-4, ls=ls, center_coord=[0,0])
    print({"objective":objective, "target angles":theta_init})
    print(f"ApproxPoint = {tApprox}, Solution = {theta} ")

