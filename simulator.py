import numpy as np
import matplotlib.pyplot as plt
class drone:
    def __init__(self,mass,velocity,position):
        self.mass=mass
        self.velocity=np.array(velocity)
        self.position=position
    def update(self,force,dt):
        acceleration=(force/self.mass)
        Vnew=self.velocity+acceleration*dt
        Xnew=self.position+(Vnew*dt)
        self.velocity=Vnew
        self.position=Xnew
# d1=drone(2,0,[0,0,0])
# target=np.array([10,10,10])
# kp=1.
# kd=2.
# position=[0,0,0]
# path=[position]
# for i in range(200):
#     error=target-d1.position
#     force=kp*error-(kd*d1.velocity)
#     d1.update(force,dt=0.1)
#     path.append(d1.position.copy())
# plt.plot(np.array(path))
# plt.show()

