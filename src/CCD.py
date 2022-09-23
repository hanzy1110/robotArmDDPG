import numpy as np

def rotateZ(theta):
    rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return rz

def translate(dx, dy, dz):
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    return t

# Forward Kinematics
# Input initial angles and length of links
# Output positions each points
def FK(angle, link):
    n_links = len(link)
    P = []
    P.append(np.eye(4))
    for a, l in zip(angle, link):
        R = rotateZ(a)
        T = translate(l, 0,0)
        P.append(P[-1].dot(R).dot(T))
    return P

def IK(target, angle, link, max_iter = 10000, err_min = 0.01):
    solved = False
    err_end_to_target = np.inf
    
    for loop in range(max_iter):
        for i in range(len(link)-1, -1, -1):
            P = FK(angle, link)
            end_to_target = target - P[-1][:3, 3]
            err_end_to_target = np.linalg.norm(end_to_target)

            # print(f"err_end_to_target: {err_end_to_target}")
            if err_end_to_target < err_min:
                solved = True
            else:
                # Calculate distance between i-joint position to end effector position
                # P[i] is position of current joint
                # P[-1] is position of end effector
                cur_to_end = P[-1][:3, 3] - P[i][:3, 3]
                cur_to_end_mag = np.linalg.norm(cur_to_end)
                cur_to_target = target - P[i][:3, 3]
                cur_to_target_mag = np.linalg.norm(cur_to_target)

                end_target_mag = cur_to_end_mag * cur_to_target_mag

                # print(f"end_target_mag: {end_target_mag}")
                if end_target_mag <= 0.0001:    
                    cos_rot_ang = 1
                    sin_rot_ang = 0
                else:
                    cos_rot_ang = (cur_to_end[0] * cur_to_target[0] + cur_to_end[1] * cur_to_target[1]) / end_target_mag
                    sin_rot_ang = (cur_to_end[0] * cur_to_target[1] - cur_to_end[1] * cur_to_target[0]) / end_target_mag

                rot_ang = np.arccos(max(-1, min(1,cos_rot_ang)))

                if sin_rot_ang < 0.0:
                    rot_ang = -rot_ang

                # Update current joint angle values
                # angle[i] = angle[i] + (rot_ang*180/np.pi)
                # print(f"rot_ang: {cos_rot_ang}")
                angle[i] = (angle[i] + rot_ang)%(2*np.pi)
                # angle[i] %= 2*np.pi

                # if angle[i] >= (2*np.pi):
                #     angle[i] = angle[i] - (2*np.pi)
                # if angle[i] < 0:
                #     angle[i] = (2*np.pi) + angle[i]
                  
        if solved:
            break
            
    return angle, err_end_to_target, solved, loop

class CCDControl:
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
        print(f"ApproxPoint = {tApprox[-1][:-1, -1]}, Solution = {angles} ")
        action = actual-angles
        return actual + alpha*action
