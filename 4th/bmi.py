import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x=np.arange(-3,3,0.25)
y=np.arange(-3,3,0.25)
XX,YY=np.meshgrid(x,y)
ZZ=XX**2+YY*2

fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(XX,YY,ZZ,rstride=1,cstride=1,cmap='hot')