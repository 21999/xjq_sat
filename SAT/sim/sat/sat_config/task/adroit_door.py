# [
#     "Arm_Translation_z_in_palm_plane",  # 0: 整个手臂向门的线性平移（z轴）
#     "Arm_Rotation_x_perpendicular_palm_plane",  # 1: 整个手臂上下角运动（x轴旋转）
#     "Arm_Rotation_y_in_palm_plane",  # 2: 整个手臂左右角运动（y轴旋转）
#     "Arm_Rotation_z_perpendicular_palm_plane",  # 3: 整个手臂滚动角运动（z轴旋转）
#     "Wrist_WRJ1_rotation_in_palm_plane",  # 4: 腕关节水平运动（桡偏/尺偏）
#     "Wrist_WRJ0_rotation_perpendicular_palm_plane",  # 5: 腕关节屈曲/伸展（垂直方向）
#     "FF_MCP_rotation_in_palm_plane",  # 6: 食指MCP关节水平运动（内收/外展）
#     "FF_MCP_rotation_perpendicular_palm_plane",  # 7: 食指MCP关节垂直运动（屈曲/伸展）
#     "FF_PIP_rotation_perpendicular_palm_plane",  # 8: 食指PIP关节屈曲/伸展
#     "FF_DIP_rotation_perpendicular_palm_plane",  # 9: 食指DIP关节运动
#     "MF_MCP_rotation_in_palm_plane",  # 10: 中指MCP关节水平运动（内收/外展）
#     "MF_MCP_rotation_perpendicular_palm_plane",  # 11: 中指MCP关节垂直运动（屈曲/伸展）
#     "MF_PIP_rotation_perpendicular_palm_plane",  # 12: 中指PIP关节屈曲/伸展
#     "MF_DIP_rotation_perpendicular_palm_plane",  # 13: 中指DIP关节运动
#     "RF_MCP_rotation_in_palm_plane",  # 14: 无名指MCP关节水平运动（内收/外展）
#     "RF_MCP_rotation_perpendicular_palm_plane",  # 15: 无名指MCP关节垂直运动（屈曲/伸展）
#     "RF_PIP_rotation_perpendicular_palm_plane",  # 16: 无名指PIP关节运动
#     "RF_DIP_rotation_perpendicular_palm_plane",  # 17: 无名指DIP关节运动
#     "LF_CMC_rotation_in_palm_plane",  # 18: 小指CMC关节运动
#     "LF_MCP_rotation_in_palm_plane",  # 19: 小指MCP关节水平运动（内收/外展）
#     "LF_MCP_rotation_perpendicular_palm_plane",  # 20: 小指MCP关节垂直运动（屈曲/伸展）
#     "LF_PIP_rotation_perpendicular_palm_plane",  # 21: 小指PIP关节屈曲/伸展
#     "LF_DIP_rotation_perpendicular_palm_plane",  # 22: 小指DIP关节运动
#     "T_CMC_rotation_in_palm_plane",  # 23: 拇指CMC关节水平运动
#     "T_CMC_rotation_perpendicular_palm_plane",  # 24: 拇指CMC关节垂直运动
#     "T_MCP_rotation_in_palm_plane",  # 25: 拇指MCP关节水平运动（内收/外展）
#     "T_MCP_rotation_perpendicular_palm_plane",  # 26: 拇指MCP关节垂直运动（屈曲/伸展）
#     "T_IP_rotation_perpendicular_palm_plane"  # 27: 拇指IP关节屈曲/伸展
# ]

# T-CMC: flexion-extension / abduction-adduction / rotation ( 0 1 2 )
# T-MCP: flexion-extension  ( 3 )
# T-IP: flexion-extension  ( 4 )
# IF-CMC: flexion-extension / abduction-adduction / rotation ( 5 6 7 )
# IF-MCP: flexion-extension / abduction-adduction / rotation ( 8 9 10 )
# IF-PIP: flexion-extension  ( 11 )
# IF-DIP: flexion-extension  ( 12 )
# MF-CMC: flexion-extension / abduction-adduction / rotation ( 13 14 15 )
# MF-MCP: flexion-extension / abduction-adduction / rotation ( 16 17 18 )
# MF-PIP: flexion-extension  ( 19 )
# MF-DIP: flexion-extension  ( 20 )
# RF-CMC: flexion-extension / abduction-adduction / rotation ( 21 22 23 )
# RF-MCP: flexion-extension / abduction-adduction / rotation ( 24 25 26 )
# RF-PIP: flexion-extension  ( 27 )
# RF-DIP: flexion-extension  ( 28 )
# LF-CMC: flexion-extension / abduction-adduction / rotation ( 29 30 31 )
# LF-MCP: flexion-extension / abduction-adduction / rotation ( 32 33 34 )
# LF-PIP: flexion-extension  ( 35 )
# LF-DIP: flexion-extension  ( 36 )

########################### ok version
adroit_door_joint_idx = [
    [4, 39],
    [4, 40],
    [4, 41],
    [4, 42],
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
# adroit_door_joint_idx = [
#     [0, 39],
#     [0, 40],
#     [0, 41],
#     [0, 42],
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
# adroit_door_joint_idx = [
#     [4, 0],
#     [4, 0],
#     [4, 0],
#     [4, 0],
#     [3, 0],
#     [3, 0],
    
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
# adroit_door_joint_idx = [
#     [4, 39],
#     [4, 40],
#     [4, 41],
#     [4, 42],
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