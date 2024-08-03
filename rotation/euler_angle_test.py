import numpy as np
from scipy.spatial.transform import Rotation as R
from rotation_tools import euler_to_quat, quat_to_euler

# test that quat_to_euler and euler_to_quat are inverses
for i in range(100):
    q = np.random.rand(4)
    q /= np.linalg.norm(q)
    e = quat_to_euler(*q)
    q2 = euler_to_quat(*e)
    assert np.allclose(q, q2)
# test that euler_to_quat and quat_to_euler are inverses
for i in range(100):
    e = np.random.rand(3)*2*np.pi
    if e[1] > np.pi:
        e[1] -= np.pi
    q = euler_to_quat(*e)
    e2 = quat_to_euler(*q)
    e2 = np.remainder(e2, 2*np.pi)
    assert np.allclose(e, e2)

# check that conversion is consistent with scipy
for i in range(100):
    q = np.random.rand(4)
    q /= np.linalg.norm(q)
    r = R.from_quat(q[[1,2,3,0]])
    e = r.as_euler('ZXZ')
    e2 = quat_to_euler(*q)
    assert np.allclose(e, e2)

for i in range(100):
    e = np.random.rand(3)*2*np.pi
    if e[1] > np.pi:
        e[1] -= np.pi
    r = R.from_euler('ZXZ', e)
    q = r.as_quat()
    q = q[[3,0,1,2]]
    q2 = euler_to_quat(*e)
    assert np.allclose(q, q2)