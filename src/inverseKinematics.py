import numpy as np
import matplotlib.pyplot as plt

def trajectoria(i, z, r, dt = 0.5):
    return r*(1+np.cos(dt * i)), r*(1+np.sin(dt * i)), z

def linea_recta(i, z, a, offset = 210, dt = 1.3):
    return -(i*dt*a) + offset, 0, z

def atan2(y,x):
    # if x == 0:
    #     return np.pi/2
    if y>0:
        return np.arctan(y/x)
    elif y<0:
        return np.arctan(y/x) + np.pi * np.sign(y)
    elif y == 0:
        return np.pi/2 * np.sign(x)

class inverseKControl:
    def __init__(self,L):
        self.L = L

    def get_angles(self, goal):
        # x,y,z, orient = goal
        x,y= goal["x"], goal['y']
        z = 0
        orient = np.pi/2

        l_1 = 0
        l_2, l_3, l_4 = self.L

        # theta_1 = atan2(y,x)
        theta_1 = 0
        
        A = x - l_4*np.cos(theta_1)*np.cos(orient)
        B = y - l_4*np.sin(theta_1)*np.cos(orient)
        C = z - l_1 - l_4*np.sin(orient)

        # print('A->', A, 'B->', B, 'C->', C)
        arg = (A**2 + B**2 + C**2 - l_2**2 - l_3**2)/(2*l_2*l_3)

        print('arg...', arg)
        theta_3 = np.arccos(arg)

        a = l_3 * np.sin(theta_3)
        b = l_2+l_3*np.cos(theta_3)
        r = np.sqrt(a**2 + b**2)

        # print('a->', a, 'b->', b, 'C->', C)
        theta_2 = np.arctan2(C,(np.sqrt(r**2-C**2))) - atan2(a,b)
        theta_4 = orient - theta_2 - theta_3

        print('theta_2', theta_2)
        # theta_1 += np.pi/2
        # theta_3 = np.pi/2 - theta_3
        # theta_4 = np.pi/2 - theta_4

        # theta_2 = theta_2+np.pi/2 if theta_2>=0 else np.pi/2 - abs(theta_2) 
        # theta_3 = theta_3+np.pi/2 if theta_3>=0 else np.pi/2 - abs(theta_3) 
        # theta_4 = theta_4+np.pi/2 if theta_4>=0 else np.pi/2 - abs(theta_4) 
        
        # conversion = 180/np.pi

        # return (theta_1* conversion)%180, (theta_2* conversion)%180, (theta_3* conversion)%180, (theta_4 * conversion)%180
        return np.array([theta_2, theta_3, theta_4])

    def get_action(self,goal, actual, alpha=0.0001):
        angles = self.get_angles(goal)
        action = angles-actual
        return actual + alpha*action

