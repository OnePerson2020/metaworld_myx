def evaluate_state(self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]) -> tuple[float, dict[str, Any]]:
    """计算当前状态的奖励和信息"""
    
    # 提取观测信息
    hand_pos = obs[:3]
    hand_quat = obs[3:7]  
    gripper_distance = obs[7]
    hand_force = obs[8:11]
    hand_torque = obs[11:14]
    obj_pos = obs[14:17]  # peg position
    obj_quat = obs[17:21] # peg quaternion
    goal_pos = obs[-3:]   # hole position
    
    # 获取插入信息
    insertion_info = self.get_insertion_info()
    insertion_depth = insertion_info["insertion_depth"]
    hole_pos = insertion_info["hole_pos"]
    hole_orientation = insertion_info["hole_orientation"]
    peg_head_pos = insertion_info["peg_head_pos"]
    
    # 计算各项奖励组件
    stage_rewards = {}
    
    # 1. 接近奖励 - 鼓励手靠近peg
    hand_to_peg_dist = np.linalg.norm(hand_pos - obj_pos)
    approach_reward = reward_utils.tolerance(
        hand_to_peg_dist, 
        bounds=(0.0, 0.0),  # 5cm内给满分
        margin=0.15,
        sigmoid='gaussian'
    )
    stage_rewards["approach"] = approach_reward * 0.1
    
    # 2. 抓取奖励 - 基于夹爪状态和接触力
    grasp_reward = 0.0
    if self.touching_main_object:
        # 夹爪闭合程度奖励
        gripper_reward = reward_utils.tolerance(
            gripper_distance,
            bounds=(0.0, 0.3),  # 期望夹爪适度闭合
            margin=0.2,
            sigmoid='gaussian'
        )
        
        # 抓取稳定性奖励（基于接触力）
        force_magnitude = np.linalg.norm(hand_force)
        force_reward = reward_utils.tolerance(
            force_magnitude,
            bounds=(0.1, 0.8),  # 适度的抓取力
            margin=0.5,
            sigmoid='gaussian'
        )
        
        grasp_reward = reward_utils.hamacher_product(gripper_reward, force_reward)
    
    stage_rewards["grasp"] = grasp_reward * 0.2
    
    # 3. 对齐奖励 - peg与hole的位置和姿态对齐
    # 位置对齐（XY平面）
    peg_to_hole_xy = np.linalg.norm((peg_head_pos - hole_pos)[:2])
    xy_alignment_reward = reward_utils.tolerance(
        peg_to_hole_xy,
        bounds=(0.0, 0.01),  # 1cm内对齐
        margin=0.05,
        sigmoid='gaussian'
    )
    
    # 姿态对齐 - peg的朝向应该与hole一致
    peg_orientation = Rotation.from_quat(obj_quat[[1,2,3,0]]).as_matrix()[:, 2]  # peg的Z轴方向
    orientation_alignment = np.dot(peg_orientation, hole_orientation)
    orientation_reward = reward_utils.tolerance(
        orientation_alignment,
        bounds=(0.8, 1.0),  # 朝向相似度
        margin=0.3,
        sigmoid='gaussian'
    )
    
    alignment_reward = reward_utils.hamacher_product(xy_alignment_reward, orientation_reward)
    stage_rewards["alignment"] = alignment_reward * 0.2
    
    # 4. 插入深度奖励 - 主要奖励
    max_insertion = 0.05  # 5cm最大插入深度
    depth_progress = min(insertion_depth / max_insertion, 1.0)
    
    # 使用平滑的插入奖励曲线
    insertion_reward = 0.0
    if insertion_depth > 0:
        # 基础插入奖励
        base_insertion = reward_utils.tolerance(
            -insertion_depth,  # 负值，因为我们要最大化插入深度
            bounds=(-max_insertion, 0.0),
            margin=0.02,
            sigmoid='linear'
        )
        
        # 深度越深，奖励越高
        depth_bonus = depth_progress ** 0.5  # 开方函数，前期奖励更明显
        
        insertion_reward = base_insertion * (1.0 + depth_bonus)
    
    stage_rewards["insertion_depth"] = insertion_reward * 0.3
    
    # 5. 动作平滑性奖励 - 减少抖动
    action_magnitude = np.linalg.norm(action)
    smoothness_reward = reward_utils.tolerance(
        action_magnitude,
        bounds=(0.0, 0.5),
        margin=0.5,
        sigmoid='gaussian'
    )
    stage_rewards["smoothness"] = smoothness_reward * 0.05
    
    # 6. 成功完成奖励
    success_reward = 0.0
    if insertion_depth >= 0.04:  # 4cm插入深度视为成功
        success_reward = 10.0
    elif insertion_depth >= 0.02:  # 2cm部分成功
        success_reward = 2.0
    
    stage_rewards["success"] = success_reward
    
    # 8. 失败惩罚
    penalty = 0.0
    
    # 如果peg掉落惩罚
    if obj_pos[2] < 0.01:  # peg掉到桌面以下
        penalty -= 1.0
    
    # 如果手离peg太远且没有抓取
    if not self.touching_main_object and hand_to_peg_dist > 0.2:
        penalty -= 0.1
    
    stage_rewards["penalty"] = penalty
    
    # 计算总奖励
    total_reward = sum(stage_rewards.values())
    
    # 构建信息字典
    info = {
        "stage_rewards": stage_rewards,
        "total_reward": total_reward,
        "insertion_depth": insertion_depth,
        "hand_to_peg_dist": hand_to_peg_dist,
        "peg_to_hole_dist": peg_to_hole_xy,
        "touching_object": self.touching_main_object,
        "gripper_distance": gripper_distance,
        "force_magnitude": np.linalg.norm(hand_force),
    }
    
    return total_reward, info
