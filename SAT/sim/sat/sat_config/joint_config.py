from sat.sat_config.task.adroit_door import adroit_door_joint_idx
from sat.sat_config.task.adroit_hammer import adroit_hammer_joint_idx
from sat.sat_config.task.adroit_pen import adroit_pen_joint_idx
from sat.sat_config.task.dexart_bucket import dexart_bucket_joint_idx
from sat.sat_config.task.dexart_faucet import dexart_faucet_joint_idx
from sat.sat_config.task.dexart_laptop import dexart_laptop_joint_idx
from sat.sat_config.task.dexart_toilet import dexart_toilet_joint_idx
from sat.sat_config.task.actionnet import actionet_joint_idx
from sat.sat_config.task.hoi4d import hoi4d_joint_idx

task_name_2_joint_desc = {
    'adroit_door': adroit_door_joint_idx,
    'adroit_hammer': adroit_hammer_joint_idx,
    'adroit_pen': adroit_pen_joint_idx,
    'dexart_bucket': dexart_bucket_joint_idx,
    'dexart_faucet': dexart_faucet_joint_idx,
    'dexart_laptop': dexart_laptop_joint_idx,
    'dexart_toilet': dexart_toilet_joint_idx,
    'actionnet_robot': actionet_joint_idx,
    'hoi4d_human': hoi4d_joint_idx,
}