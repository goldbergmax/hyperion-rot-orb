import rebound
import numpy as np
import matplotlib.pyplot as plt

sim = rebound.Simulation()
sim.units = ('d', 'km', 'msaturn')
sim.add('Saturn')
sim.add('Titan')
sim.add('Hyperion')
sim.move_to_com()
sim.integrator = 'whfast'
ps = sim.particles
sim.dt = ps[1].P/20
times = np.linspace(0, 40*365.25, 100)
orbit = np.zeros((len(times), 3))
phi0 = ps[1].pomega - ps[2].pomega
for t in times:
    sim.integrate(t)
    orbit[times==t] = ps[2].a, ps[2].e, ps[2].pomega
    print(t/365.25, ps[2].e)

plt.plot(times/365.25, orbit[:,1])
plt.plot(times/365.25, 0.104 + 0.024*np.cos(2*np.pi*times/6850.0 + phi0))
plt.xlabel('Time (years)')
plt.ylabel('Hyperion eccentricity')
plt.savefig('../figs/hyperion_eccentricity.png')
