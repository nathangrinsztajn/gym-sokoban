import numpy as np

def _calc_ball_on_walls(s):
    balls_pos = (s == 4) + (s == 3)

    collision_pos1 = balls_pos & (np.roll(s, 1, axis=0) == 0)
    collision_pos2 = balls_pos & (np.roll(s, -1, axis=0) == 0)
    collision_pos3 = balls_pos & (np.roll(s, 1, axis=1) == 0)
    collision_pos4 = balls_pos & (np.roll(s, -1, axis=1) == 0)

    collision_pos = collision_pos1 + collision_pos2 + collision_pos3 + collision_pos4
    collisions = np.sum(collision_pos)

    coin_pos = balls_pos & (np.roll(s, 1, axis=0) == 0) & (np.roll(s, 1, axis=1) == 0)
    coin_pos += balls_pos & (np.roll(s, -1, axis=0) == 0) & (np.roll(s, 1, axis=1) == 0)
    coin_pos += balls_pos & (np.roll(s, 1, axis=0) == 0) & (np.roll(s, -1, axis=1) == 0)
    coin_pos += balls_pos & (np.roll(s, -1, axis=0) == 0) & (np.roll(s, -1, axis=1) == 0)
    coins = np.sum(coin_pos)

    return collisions, coins, collision_pos.astype(int), coin_pos.astype(int), [collision_pos1.astype(int),
                                                                                collision_pos2.astype(int),
                                                                                collision_pos3.astype(int),
                                                                                collision_pos4.astype(int)]


def _check_reversible_push(room_state, box_coordinates, ax=0):
    x, y = box_coordinates

    if ax == 0:
        array_wall = room_state[x - 1, :]
        array_box = room_state[x, :]
        pos_fix = x

    elif ax == 1:
        array_wall = room_state[x + 1, :]
        array_box = room_state[x, :]
        pos_fix = x

    elif ax == 2:
        array_wall = room_state[:, y - 1]
        array_box = room_state[:, y]
        pos_fix = y

    else:
        array_wall = room_state[:, y + 1]
        array_box = room_state[:, y]
        pos_fix = y

    is_wall = np.nonzero(array_box == 0)[0]
    is_wall_before = is_wall[is_wall < pos_fix]
    is_wall_after = is_wall[is_wall > pos_fix]

    is_void = np.nonzero(array_wall > 0)[0]
    is_void_before = is_void[is_void < pos_fix]
    is_void_after = is_void[is_void > pos_fix]

    left_side = False
    if len(is_void_before) == 0:
        left_side = True
    if len(is_void_before) > 0:
        if np.max(is_void_before) <= np.max(is_wall_before):
            left_side = True

    right_side = False
    if len(is_void_after) == 0:
        right_side = True
    if len(is_void_after) > 0:
        if np.min(is_void_after) >= np.min(is_wall_after):
            right_side = True
    return right_side and left_side


def proxy_oracle(s1, s2):
    """
    return 1 if irreversible, else 0
    """
    collisions1, coins1, collision_pos1, coin_pos1, collisionL1 = _calc_ball_on_walls(s1)
    collisions2, coins2, collision_pos2, coin_pos2, collisionL2 = _calc_ball_on_walls(s2)

    if coins2 > coins1:
        return 1

    # on a wall: reversible ?
    for i, (collision_ax1, collision_ax2) in enumerate(zip(collisionL1, collisionL2)):
        new_box_pos = np.nonzero((collision_ax2 - collision_ax1) > 0)
        for box_coordinates in zip(new_box_pos[0], new_box_pos[1]):
            rev = _check_reversible_push(s2, box_coordinates, ax=i)
            if rev == 1:
                return 1
    return 0
