# [
#     "Arm_Rotation_x_perpendicular_palm_plane",  # 0: 整个手臂上下角运动（x轴旋转）
#     "Arm_Rotation_y_in_palm_plane",  # 1: 整个手臂左右角运动（y轴旋转）
#     "Wrist_WRJ1_rotation_in_palm_plane",  # 2: 腕关节水平运动（桡偏/尺偏）
#     "Wrist_WRJ0_rotation_perpendicular_palm_plane",  # 3: 腕关节屈曲/伸展（垂直方向）
#     "FF_MCP_rotation_in_palm_plane",  # 4: 食指MCP关节水平运动（内收/外展）
#     "FF_MCP_rotation_perpendicular_palm_plane",  # 5: 食指MCP关节垂直运动（屈曲/伸展）
#     "FF_PIP_rotation_perpendicular_palm_plane",  # 6: 食指PIP关节屈曲/伸展
#     "FF_DIP_rotation_perpendicular_palm_plane",  # 7: 食指DIP关节运动
#     "MF_MCP_rotation_in_palm_plane",  # 8: 中指MCP关节水平运动（内收/外展）
#     "MF_MCP_rotation_perpendicular_palm_plane",  # 9: 中指MCP关节垂直运动（屈曲/伸展）
#     "MF_PIP_rotation_perpendicular_palm_plane",  # 10: 中指PIP关节屈曲/伸展
#     "MF_DIP_rotation_perpendicular_palm_plane",  # 11: 中指DIP关节运动
#     "RF_MCP_rotation_in_palm_plane",  # 12: 无名指MCP关节水平运动（内收/外展）
#     "RF_MCP_rotation_perpendicular_palm_plane",  # 13: 无名指MCP关节垂直运动（屈曲/伸展）
#     "RF_PIP_rotation_perpendicular_palm_plane",  # 14: 无名指PIP关节运动
#     "RF_DIP_rotation_perpendicular_palm_plane",  # 15: 无名指DIP关节运动
#     "LF_CMC_rotation_in_palm_plane",  # 16: 小指CMC关节运动
#     "LF_MCP_rotation_in_palm_plane",  # 17: 小指MCP关节水平运动（内收/外展）
#     "LF_MCP_rotation_perpendicular_palm_plane",  # 18: 小指MCP关节垂直运动（屈曲/伸展）
#     "LF_PIP_rotation_perpendicular_palm_plane",  # 19: 小指PIP关节屈曲/伸展
#     "LF_DIP_rotation_perpendicular_palm_plane",  # 20: 小指DIP关节运动
#     "T_CMC_rotation_in_palm_plane",  # 21: 拇指CMC关节水平运动
#     "T_CMC_rotation_perpendicular_palm_plane",  # 22: 拇指CMC关节垂直运动
#     "T_MCP_rotation_in_palm_plane",  # 23: 拇指MCP关节水平运动（内收/外展）
#     "T_MCP_rotation_perpendicular_palm_plane",  # 24: 拇指MCP关节垂直运动（屈曲/伸展）
#     "T_IP_rotation_perpendicular_palm_plane"  # 25: 拇指IP关节屈曲/伸展
# ]

adroit_hammer_joint_idx = [
    [4, 40],
    [4, 41],
    [3, 41],
    [3, 40],
    
    [2, 9],
    [2, 8],
    [2, 11],
    [2, 12],
    [2, 17],
    [2, 16],
    [2, 19],
    [2, 20],
    [2, 25],
    [2, 24],
    [2, 27],
    [2, 28],
    [2, 29], # mainly flexion-extension, hard to determine
    [2, 33],
    [2, 32],
    [2, 35],
    [2, 36],
    [2, 2],
    [2, 1],
    [2, 1], # redundant 
    [2, 3],
    [2, 4]
]