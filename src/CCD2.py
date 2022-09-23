import numpy as np

def shoulder(angle, link):
    return np.array( [ link[0]*np.cos(angle[0]), link[0]*np.sin(angle[0]), 0, 1 ] ).reshape(-1,1)

def elbow(angle, link):
    return np.array( [ link[0]*np.cos(angle[0])+link[1]*np.cos(angle[:2].sum()),
                       link[0]*np.sin(angle[0])+link[1]*np.sin(angle[:2].sum()),
                       0,
                       1 ] ).reshape(-1,1)

def wrist(angle, link):
    return np.array( [ link[0]*np.cos(angle[0])+link[1]*np.cos(angle[:2].sum())+link[2]*np.cos(angle.sum()),
                       link[0]*np.sin(angle[0])+link[1]*np.sin(angle[:2].sum())+link[2]*np.sin(angle.sum()),
                       0,
                       1 ] ).reshape(-1,1)

arm_funcs = {0:shoulder, 1:elbow, 2:wrist}

# Forward Kinematics
# Input initial angles and length of links
# Output positions each points
def FK(angle, link):
    return np.concatenate([arm_funcs[linkIDX](angle, link) for linkIDX in range(len(link))], axis=1)

def dist(P, goal):
    return np.linalg.norm(P-goal)

def secanteIter(P, P_prev, angle, angle_prev, goal):
    # import pdb; pdb.set_trace()
    return angle - dist(P, goal) * (angle-angle_prev)/(dist(P, goal)-dist(P_prev, goal))

def IK(target, angle, link, max_iter = 10000, err_min = 0.01):
    solved = False
    err_end_to_target = np.inf
    
    angle_prev = angle*0.01

    for i in range(len(link)-1, -1, -1):

        for loop in range(max_iter):

            P = FK(angle, link)
            P_prev = FK(angle_prev, link)

            end_to_target = target - P[:-1, -1]
            err_end_to_target = np.linalg.norm(end_to_target)/np.linalg.norm(target)

            angle_new = secanteIter(P[:-1, i], P_prev[:-1, i], angle[i], angle_prev[i], target)

            # Update current joint angle values
            if err_end_to_target<=err_min:
                solved=True
            elif np.abs(angle_new-angle[i])<=err_min:
                solved=True
            else:
                angle_prev = angle
                angle[i] = angle_new%(2*np.pi)

        if solved:
            break
            
    return angle, err_end_to_target, solved, loop

class CCDControl2:
    def __init__(self, ls, center_coord):
        self.ls = ls
        self.center_coord = center_coord

    def correctGoal(self, goal):
        g = np.array([goal['x'], goal['y'], 0])
        g[0] -= self.center_coord[0]
        g[1] -= self.center_coord[1]
        return g

    def get_angles(self, goal, actual):
        angle, err_end_to_target, solved, loop = IK(goal, actual, self.ls)
        tApprox = FK(angle, self.ls)
        print({'Error: ': err_end_to_target, 'iters: ':loop, 'solved':solved})
        return angle, tApprox

    def get_action(self,goal, actual, alpha=0.1):
        goal = self.correctGoal(goal)
        # actualDegs = np.rad2deg(actual)
        angles, tApprox = self.get_angles(goal, actual)
        # angles = np.deg2rad(np.array(angles))
        print(f"Goal {goal}")
        print(f"ApproxPoint = {tApprox[:-1,-1]}, Solution = {angles} ")
        action = actual-angles
        return actual - alpha*action
