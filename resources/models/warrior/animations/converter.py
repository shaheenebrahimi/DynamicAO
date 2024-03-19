import numpy as np

to_rad = np.pi / 180

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    magnitude = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
    return [qx/magnitude, qy/magnitude, qz/magnitude, qw/magnitude]

# input: z, y, x
r = (40*to_rad, 20*to_rad, 0*to_rad)
print(euler_to_quaternion(r))
print(0.0593912**2 + 0.1631759**2 + 0.3368241**2 + 0.9254166**2)
