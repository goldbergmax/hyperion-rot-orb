A library to analyze the rotation and orbit of Hyperion. The code is split into two indepedent parts, one for the rotation and one for the orbit.
If you use this code, please cite: **Goldberg & Batygin, in prep.**


---

#### Rotation
Requirements: numpy, scipy, matplotlib, rebound (to efficiently solve Kepler's equation)

rotation_tools.RotationalState and rotation_tools.RotationSimulation are general classes to solve Euler's rotation equations under a planetary torque. They support both Euler angles or quaternions for the initial conditions, outputs, and the integration itself. See hyp_rotation.py for a particular implementation for Hyperion.

#### Orbit
Requirements: numpy, scipy, rebound, reboundx

hyp_capture.Simulation is a class to simulate the capture of Hyperion into its current orbit. It uses rebound to integrate the orbit, and reboundx to include the tidal effects of Saturn and tidal dissipation inside Hyperion. The HyperionDamping class provides multiple prescriptions for the dissipation inside Hyperion as a function of orbital eccentricity. TitanMigration is similar for the migration of Titan due to tides raised on Saturn. 

![tumbling](https://github.com/goldbergmax/hyperion-rot-orb/assets/46541633/169e623d-e602-47cf-97d8-5bc4812ae6bc)
