import numpy as np


#
# def x_vel_tracking(x_vel, command):
#     x_vel_error = x_vel - command
#     return np.exp(-np.square(x_vel_error))
#
# def y_vel_tracking(y_vel, command):
#     y_vel_error = y_vel - command
#     return np.exp(-0.1*np.square(y_vel_error))
#
#
# def ang_vel_tracking(ang_vel, command):
#     ang_vel_error = ang_vel - command  # yaw
#     return np.exp(-np.square(ang_vel_error))
#
#
# def base_motion(lin_vel, ang_vel):
#     # return np.square(lin_vel[2]) + 0.5 * np.sum(np.square(ang_vel[:2]))
#     return 0.2 * np.fabs(ang_vel[0]) + 0.2 * np.fabs(ang_vel[1]) + 0.8 * np.square(lin_vel[2])
#
#
# def base_orientation(gravity_vec):
#     return abs(gravity_vec[0])
#
#
# def torque_regularization(torque):
#     return np.sum(np.square(torque))

def lin_vel_tracking(lin_vel, command):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = np.sum(np.square(command[:2] - lin_vel[:2]))
    return np.exp(-lin_vel_error / 0.25)


def ang_vel_tracking(ang_vel, command):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = np.square(command[2] - ang_vel[2])
    return np.exp(-ang_vel_error / 0.25)


def z_vel_penalty(lin_vel):
    # Penalize z axis base linear velocity
    return np.square(lin_vel[2])


def roll_pitch_penalty(ang_vel):
    # Penalize xy axes base angular velocity
    return np.sum(np.square(ang_vel[:2]))


def base_orientation_penalty(projected_gravity):
    # Penalize non flat base orientation
    return np.sum(np.square(projected_gravity[:2]))


def torque_penalty(torque):
    return np.sum(np.square(torque))

def foot_lateral_distance_penalty(rfoot_poses, lfoot_poses): #TODO: check if this is correct
    ''' 
    Get the closest distance between the two feet and make it into a penalty. The given points are five key points in the feet.
    Args:
        rfoot_poses: [3,5]
        lfoot_poses: [3,5]
    '''
    assert rfoot_poses.shape == (3, 5) and lfoot_poses.shape == (3, 5), 'foot poses should be 5x3'
    
    distance0 = np.abs(rfoot_poses[1,0] - lfoot_poses[1,0])
    distance1 = np.abs(rfoot_poses[1,4] - lfoot_poses[1,3])
    distance2 = np.abs(rfoot_poses[1,3] - lfoot_poses[1,4])
    distances = np.array([distance0, distance1, distance2])
    closest_distance = np.min(distances)
    
    # return (closest_distance<0.27) * closest_distance
    return closest_distance<0.13

def swing_foot_fix_penalty(lfoot_grf, rfoot_grf, action):
    ''' penalize if the toe joint changes from its fixed position in swing phase '''
    # TODO: check if contact check is correct
    lfoot_penalty = (lfoot_grf < 1) * np.sum(np.square(action[4:6]))
    rfoot_penalty = (rfoot_grf < 1) * np.sum(np.square(action[10:12]))
    return lfoot_penalty + rfoot_penalty