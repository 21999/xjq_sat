# [
#     "Wrist_WRJ1_rotation_in_palm_plane",  # 0: 腕关节水平运动（桡偏/尺偏）
#     "Wrist_WRJ0_rotation_perpendicular_palm_plane",  # 1: 腕关节屈曲/伸展（垂直方向）
#     "FF_MCP_rotation_in_palm_plane",  # 2: 食指MCP关节水平运动（内收/外展）
#     "FF_MCP_rotation_perpendicular_palm_plane",  # 3: 食指MCP关节垂直运动（屈曲/伸展）
#     "FF_PIP_rotation_perpendicular_palm_plane",  # 4: 食指PIP关节屈曲/伸展
#     "FF_DIP_rotation_perpendicular_palm_plane",  # 5: 食指DIP关节运动
#     "MF_MCP_rotation_in_palm_plane",  # 6: 中指MCP关节水平运动（内收/外展）
#     "MF_MCP_rotation_perpendicular_palm_plane",  # 7: 中指MCP关节垂直运动（屈曲/伸展）
#     "MF_PIP_rotation_perpendicular_palm_plane",  # 8: 中指PIP关节屈曲/伸展
#     "MF_DIP_rotation_perpendicular_palm_plane",  # 9: 中指DIP关节运动
#     "RF_MCP_rotation_in_palm_plane",  # 10: 无名指MCP关节水平运动（内收/外展）
#     "RF_MCP_rotation_perpendicular_palm_plane",  # 11: 无名指MCP关节垂直运动（屈曲/伸展）
#     "RF_PIP_rotation_perpendicular_palm_plane",  # 12: 无名指PIP关节运动
#     "RF_DIP_rotation_perpendicular_palm_plane",  # 13: 无名指DIP关节运动
#     "LF_CMC_rotation_in_palm_plane",  # 14: 小指CMC关节运动
#     "LF_MCP_rotation_in_palm_plane",  # 15: 小指MCP关节水平运动（内收/外展）
#     "LF_MCP_rotation_perpendicular_palm_plane",  # 16: 小指MCP关节垂直运动（屈曲/伸展）
#     "LF_PIP_rotation_perpendicular_palm_plane",  # 17: 小指PIP关节屈曲/伸展
#     "LF_DIP_rotation_perpendicular_palm_plane",  # 18: 小指DIP关节运动
#     "T_CMC_rotation_in_palm_plane",  # 19: 拇指CMC关节水平运动
#     "T_CMC_rotation_perpendicular_palm_plane",  # 20: 拇指CMC关节垂直运动
#     "T_MCP_rotation_in_palm_plane",  # 21: 拇指MCP关节水平运动（内收/外展）
#     "T_MCP_rotation_perpendicular_palm_plane",  # 22: 拇指MCP关节垂直运动（屈曲/伸展）
#     "T_IP_rotation_perpendicular_palm_plane"  # 23: 拇指IP关节屈曲/伸展
# ]

########################### ok version
adroit_pen_joint_idx = [
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


########################### abla, wo, e
# adroit_pen_joint_idx = [
#     [0, 41],
#     [0, 40],
    
#     [0, 9],
#     [0, 8],
#     [0, 11],
#     [0, 12],
#     [0, 17],
#     [0, 16],
#     [0, 19],
#     [0, 20],
#     [0, 25],
#     [0, 24],
#     [0, 27],
#     [0, 28],
#     [0, 29], # mainly flexion-extension, hard to determine
#     [0, 33],
#     [0, 32],
#     [0, 35],
#     [0, 36],
#     [0, 2],
#     [0, 1],
#     [0, 1], # redundant 
#     [0, 3],
#     [0, 4]
# ]


########################### abla, wo, f
# adroit_pen_joint_idx = [
#     [4, 0],
#     [4, 0],
    
#     [2, 1],
#     [2, 0],
#     [2, 0],
#     [2, 0],
#     [2, 1],
#     [2, 0],
#     [2, 0],
#     [2, 0],
#     [2, 1],
#     [2, 0],
#     [2, 0],
#     [2, 0],
#     [2, 0], # mainly flexion-extension, hard to determine
#     [2, 1],
#     [2, 0],
#     [2, 0],
#     [2, 0],
#     [2, 2],
#     [2, 1],
#     [2, 1], # redundant 
#     [2, 0],
#     [2, 0]
# ]


# ########################### abla, wo, a
# adroit_pen_joint_idx = [
#     [3, 41],
#     [3, 40],
    
#     [2, 8],
#     [2, 8],
#     [2, 11],
#     [2, 12],
#     [2, 16],
#     [2, 16],
#     [2, 19],
#     [2, 20],
#     [2, 24],
#     [2, 24],
#     [2, 27],
#     [2, 28],
#     [2, 29], # mainly flexion-extension, hard to determine
#     [2, 33],
#     [2, 32],
#     [2, 35],
#     [2, 36],
#     [2, 0],
#     [2, 0],
#     [2, 0], # redundant 
#     [2, 3],
#     [2, 4]
# ]