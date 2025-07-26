import numpy as np
from scipy.spatial.transform import Rotation as R

# 方法1：使用 from_euler (角度单位为弧度)
rotation_deg = 90
rotation_rad = np.radians(rotation_deg)
rot_euler_xyz = R.from_euler('xyz', [0, rotation_rad, 0], degrees=True).as_euler('xyz')

print(rot_euler_xyz)  # 获取旋转矩阵
# 方法2：使用 from_euler (直接指定角度和单位)
# rot_euler_xyz_2 = R.from_euler('xyz', [0, rotation_deg, 0], degrees=True)

# # 获取旋转矩阵
# rotation_matrix = rot_euler_xyz.as_matrix()

# # 获取四元数 (scalar last format: x, y, z, w)
# quaternion_xyzw = rot_euler_xyz.as_quat()

# # 获取欧拉角 (确认)
# euler_xyz_rad = rot_euler_xyz.as_euler('xyz')
# euler_xyz_deg = rot_euler_xyz.as_euler('xyz', degrees=True)

# print("Rotation Matrix:")
# print(rotation_matrix)
# print("\nQuaternion (x, y, z, w):")
# print(quaternion_xyzw)
# print("\nEuler Angles (XYZ, radians):")
# print(euler_xyz_rad)
# print("\nEuler Angles (XYZ, degrees):")
# print(euler_xyz_deg)

# # 如果你需要 'wxyz' 格式的四元数
# quaternion_wxyz = np.roll(quaternion_xyzw, 1)
# print("\nQuaternion (w, x, y, z):")
# print(quaternion_wxyz)` `