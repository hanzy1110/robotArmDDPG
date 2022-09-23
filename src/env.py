import numpy as np
import pyglet

def shoulder(angle, link):
    return np.array( [ link[0]*np.cos(angle[0]), link[0]*np.sin(angle[0])] )

def elbow(angle, link):
    return np.array( [ link[0]*np.cos(angle[0])+link[1]*np.cos(angle[:-1].sum()),
                       link[0]*np.sin(angle[0])+link[1]*np.sin(angle[:-1].sum()),])

def wrist(angle, link):
    return np.array( [ link[0]*np.cos(angle[0]) + link[1]*np.cos(angle[:-1].sum()) + link[2]*np.cos(angle.sum()),
                       link[0]*np.sin(angle[0]) + link[1]*np.sin(angle[:-1].sum()) + link[2]*np.sin(angle.sum()),])

def wrist_render(angle, link):
    return np.array( [ link[0]*np.cos(angle[0]) + link[1]*np.cos(angle[1:].sum()) + link[2]*np.cos(angle[:-1].sum()),
                      link[0]*np.sin(angle[0]) + link[1]*np.sin(angle[1:].sum()) + link[2]*np.sin(angle[:-1].sum()),])


def compareGoal(finger, goal):
    print(f"finger = {finger}")
    print(f"goal = {goal}")

    first = goal['x'] - goal['l']/2 < finger[0] < goal['x'] + goal['l']/2
    second = goal['y'] - goal['l']/2 < finger[1] < goal['y'] + goal['l']/2

    print(f"first = {first}")
    print(f"second = {second}")
    return first, second

class ArmEnv(object):
    viewer = None
    dt = .01    # refresh rate
    # action_bound = [-1, 1]
    action_bound = [0, 2*np.pi]
    # goal = {'x': 300., 'y': 300., 'l': 40}
    state_dim = 9
    action_dim = 2

    def __init__(self, goal):
        self.n_arms = 3
        self.goal=goal

        self.center_coord = np.array([500, 500])

        self.goal['x'] += self.center_coord[0]
        self.goal['y'] += self.center_coord[1]
        self.arm_info = np.zeros(
            self.n_arms, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0

    def step(self, action):
        done = False
        # action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action *  self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord
        # a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        # a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a2 end (x2, y2)
        a1xy_ = shoulder(self.arm_info['r'], self.arm_info['l']) + a1xy
        a2xy_ = elbow(self.arm_info['r'], self.arm_info['l']) + a1xy
        finger = wrist(self.arm_info['r'],self.arm_info['l']) + a1xy
        # finger = np.array([np.cos(a3r + a4r), np.sin(a3r + a4r)]) * a4l + a3xy_  # a2 end (x2, y2)

        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)

        first, second = compareGoal(finger, self.goal)
        # done and reward
        if first and second:
            r += 1.
            self.on_goal += 1
            print(f"On goal: {self.on_goal}")
            if self.on_goal >= 40000:
                done = True
        else:
            # pass
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        # self.goal['x'] = np.random.rand()*400.
        # self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(self.n_arms)
        # self.arm_info['r'] = 1 * np.pi/2
        self.on_goal = 0
        # (a1l, a2l) = self.arm_info['l']  # radius, arm length
        # (a1r, a2r) = self.arm_info['r']  # radian, angle
        # a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        # a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        # finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord
        # a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        # a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a2 end (x2, y2)
        a1xy_ = shoulder(self.arm_info['r'], self.arm_info['l']) + a1xy
        a2xy_ = elbow(self.arm_info['r'], self.arm_info['l']) + a1xy
        finger = wrist(self.arm_info['r'],self.arm_info['l']) + a1xy
        # finger = np.array([np.cos(a3r + a4r), np.sin(a3r + a4r)]) * a4l + a3xy_  # a2 end (x2, y2)

        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal, self.center_coord)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal, center_coord):
        self.center_coord = center_coord
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=2*self.center_coord[0], height=2*self.center_coord[1],
                                     resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (250, 22, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (100, 86, 86) * 4,))
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [350, 350,              # location
                     300, 360,
                     400, 460,
                     400, 450]), ('c3B', (10, 32, 86) * 4,))
        # self.arm4 = self.batch.add(
        #     4, pyglet.gl.GL_QUADS, None,
        #     ('v2f', [400, 450,              # location
        #              400, 460,
        #              500, 560,
        #              500, 550]), ('c3B', (249, 86, 86) * 4,))



    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        # (a1l, a2l) = self.arm_info['l']     # radius, arm length
        # (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)

        # (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        # (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        # a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        # a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # a3xy_ = np.array([np.cos(a2r + a3r), np.sin(a2r + a3r)]) * a3l + a2xy_  # a2 end (x2, y2)
        # a4xy_ = np.array([np.cos(a3r + a4r), np.sin(a3r + a4r)]) * a4l + a3xy_  # a2 end (x2, y2)

        a1xy_ = shoulder(self.arm_info['r'], self.arm_info['l']) + a1xy
        a2xy_ = elbow(self.arm_info['r'], self.arm_info['l']) + a1xy
        a3xy_ = wrist_render(self.arm_info['r'], self.arm_info['l']) + a1xy

        a1tr, a2tr, a3tr = np.pi / 2 - self.arm_info['r'][0], \
            np.pi / 2 - self.arm_info['r'][0:2].sum(), \
            np.pi / 2 - self.arm_info['r'][1:].sum(), \
            # np.pi / 2 - self.arm_info['r'][2:].sum()

        #base
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        #punta
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        #base
        xy11_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        #punta
        xy21 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc

        #base
        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        #punta
        xy31 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc

        #base
        #xy31_ = a3xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc
        #xy32_ = a3xy_ + np.array([np.cos(a4tr), -np.sin(a4tr)]) * self.bar_thc
        #punta
        #xy41 = a4xy_ + np.array([np.cos(a4tr), -np.sin(a4tr)]) * self.bar_thc
        #xy42 = a4xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc


        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))
        # self.arm4.vertices = np.concatenate((xy31_, xy32_, xy41, xy42))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        # pass
        self.goal_info['x'] = x
        self.goal_info['y'] = y


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
