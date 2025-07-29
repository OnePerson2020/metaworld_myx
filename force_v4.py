            if force_magnitude > 10:
                # 1. 获取peg当前的姿态（从mujoco四元数转换为scipy Rotation对象）
                peg_current_rotation = Rotation.from_quat(o_d["peg_rot"])
                # 2. 定义在peg局部坐标系下的力臂向量
                lever_arm_local = np.array([-1, 0.0, 0.0])
                # 3. 将力臂向量从局部坐标系转换到世界坐标系
                lever_arm_world = peg_current_rotation.apply(lever_arm_local)
                corrective_torque_vector = np.cross(lever_arm_world, force_vector)
                # 5. 限制修正速度，保证稳定性
                speed = np.deg2rad(1) # 最大修正角速度
                torque_magnitude = np.linalg.norm(corrective_torque_vector)
                if torque_magnitude > 1e-6: # 避免除以零
                    unit_torque_axis = corrective_torque_vector / torque_magnitude
                    # 如果计算出的旋转速度超过上限，则使用上限速度
                    if torque_magnitude > speed:
                        increment_rotvec = unit_torque_axis * speed
                    else:
                        increment_rotvec = corrective_torque_vector
                    r_correction = Rotation.from_rotvec(increment_rotvec)
                    self.ini_r = r_correction * self.ini_r
                    desir_pos = pos_curr
                    
                    print(f"Force Detected: {force_magnitude:.2f} N. Applying rotational correction.")
                    print(f"Corrected Target Euler: {r_correction.as_euler('xyz', degrees=True)}")
                    
                    
# ppo_test/sawyer_xyz_env.py

# ... (在 SawyerXYZEnv 类中) ...
def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
    # ... (前面的速度计算等逻辑保持不变) ...
    
    # --- 导纳控制逻辑 ---
    pos_deviation = np.zeros(3)
    r_correct = Rotation.from_quat([0, 0, 0, 1])

    if hasattr(self, '_get_pegHead_force') and callable(getattr(self, '_get_pegHead_force')):
        try:
            # 1. 获取外部力 F_ext
            force = self._get_pegHead_force()

            # 2. 位置导纳: (逻辑不变，刚度由初始化参数决定)
            damping_force = self.admittance_damping * tcp_vel
            net_force = force - damping_force
            safe_stiffness = np.where(self.admittance_stiffness == 0, 1e-6, self.admittance_stiffness)
            pos_deviation = net_force / safe_stiffness

            # 3. 旋转导纳 (关键修改点)
            # 我们只希望响应 YZ 平面的力和姿态
            # - 力 Fy (force[1]) 会导致绕 X 轴的旋转 (翻滚 Roll)，这是 YZ 平面内的姿态调整。
            # - 我们不再响应 Fx (force[0])，以保持 X 轴指向的刚性。
            # highlight-start
            rotation_vec = (1.0 / self.rotational_stiffness) * np.array([force[1], 0, 0])
            # highlight-end
            if np.linalg.norm(rotation_vec) > 1e-4:
                r_correct = Rotation.from_rotvec(rotation_vec)

        except Exception:
            pass # 如果出错则不修正
    
    # ... (后面的应用控制逻辑保持不变) ...