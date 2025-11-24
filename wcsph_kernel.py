import warp as wp
import numpy as np


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x

@wp.func
def get_cubic(r_norm : float, radius : float):
    """
    Cubic spline smoothing kernel.
    """
    res = 0.0
    h = radius
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = square(q)
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = 2.0 * k * cube((1.0 - q))
            # res = cube(res)
    return res

@wp.func
def get_cubic_derivative(r: wp.vec3, smoothing_length: float):
    """
        Derivative of cubic spline smoothing kernel.
    """
    # res = ti.Vector.zero(gs.ti_float, 3)
    res = wp.vec3(0.0, 0.0, 0.0)
    

    # r_norm = radius.norm()
    r_norm = wp.length(r)
    h = smoothing_length
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
        else:
            q_term = 1.0 - q
            res = -6.0 * k * q_term * q_term * grad_q
    return res

@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float, volume: float):
    # calculate distance

    distance = wp.sqrt(wp.dot(xyz, xyz))
    # distance = wp.length(xyz)
    return volume * get_cubic(distance, smoothing_length)

# @wp.func
# def non_pressure(mass: float, gamma:) :


@wp.kernel
def rho(
        grid_id: wp.uint64,
        rho_0: float,
        volume: float, 
        smoothing_length: float,
        particle_rho: wp.array(dtype=float), 
        particle_x: wp.array(dtype=wp.vec3),
    ) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    
    x = particle_x[i]

    rho_temp = float(0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors :
        distance = x - particle_x[index]
        r_norm = wp.length(distance)
        mass_nei = rho_0 * volume
        rho_temp += mass_nei * get_cubic(r_norm, smoothing_length)
    particle_rho[i] = rho_temp


@wp.kernel
def pressure(
    stiffness: float,
    exp: float,
    rho_0: float,
    particle_rho: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
):
    tid = wp.tid()
    pressure[tid] = stiffness * (wp.pow(particle_rho[tid] / rho_0, exp) - 1.0)


@wp.func
def cal_acc_with_non_pressure(
    a: wp.vec3,
    rho_nei: float,
    vel: wp.vec3,
    vel_nei: wp.vec3,
    mass: float,
    mu: float,
    gamma: float,
    d_current_nei: wp.vec3,
    e_dist: float,
    distance: float,
    smoothing_length: float,
    
):
    a -= gamma / mass * mass * d_current_nei * get_cubic(e_dist, smoothing_length)
    v_current_nei = wp.dot((vel - vel_nei), d_current_nei)
    d = 2.0 * (3.0 + 2.0)

    f_v = (d 
        * mu
        * (mass / rho_nei) 
        * v_current_nei 
        / (distance * distance + 0.01 * (smoothing_length * smoothing_length)) 
        * get_cubic_derivative(d_current_nei, smoothing_length)
    )
    a += f_v

    return a
@wp.func
def cal_acc_with_pressure(
    a: wp.vec3,
    volume: float,
    d: wp.vec3,
    pressure: float,
    pressure_nei: float,
    rho_0: float,
    rho: float,
    rho_nei: float,
    smoothing_length: float,
):
    dp_i = pressure / (rho * rho)
    # rho_nei = rho_nei * rho_0 / rho_0
    dp_nei = pressure_nei / (rho_nei * rho_nei)
    a += (
        -rho_0
        * volume
        * (dp_i + dp_nei)
        * get_cubic_derivative(d, smoothing_length)
    )
    return a


@wp.kernel
def acceleration(
    grid_id: wp.uint64,
    rho_0: float,
    gamma: float,
    mu: float,
    mass: float,
    particle_size: float,
    volume: float,
    smoothing_length: float,
    gravity: float,
    particle_x : wp.array(dtype=wp.vec3),
    particle_v : wp.array(dtype=wp.vec3),
    particle_rho : wp.array(dtype=float),
    particle_a : wp.array(dtype=wp.vec3),
    particle_pressure: wp.array(dtype=float),
) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    rho = particle_rho[i]
    x = particle_x[i]
    a = particle_a[i]
    # data from material
    pressure = particle_pressure[i]
    acc = wp.vec3(0.0, 0.0, 0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)

    for index in neighbors :
        if index != i :
            nei_x = particle_x[index]
            dir_current_nei = x - nei_x
            # distance = wp.sqrt(wp.dot(x, nei_x))
            distance = wp.length(dir_current_nei)
            e_dist = wp.max(distance, particle_size)
            rho_nei = particle_rho[index]
            pressure_nei = particle_pressure[index]
            # non pressure acceleration
            acc = cal_acc_with_non_pressure(
                acc, 
                rho_nei, 
                particle_v[i], particle_v[index],  # velocity of current and nei particles
                mass, # mass of current and nei particles
                mu,
                gamma,
                dir_current_nei, 
                e_dist, 
                distance, 
                smoothing_length)
            
            # pressure acceleration
            acc = cal_acc_with_pressure(
                acc,
                volume,
                dir_current_nei, # position of current and nei particles
                pressure, pressure_nei,# pressure of current and nei particles
                rho_0, # rest density of current and nei particles
                rho, rho_nei, # density of current and nei particles
                smoothing_length)

    particle_a[i] = acc + wp.vec3(0.0, 0.0, gravity)
    # particle_a[i] = acc

@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), 
    particle_distance: float, 
    width: float, height: float, length: float
):
    tid = wp.tid()

    particle_spacing = particle_distance
    # grid size
    nr_x = wp.int32(width / particle_spacing)
    nr_y = wp.int32(height / particle_spacing)
    nr_z = wp.int32(length / particle_spacing)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = particle_distance * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * particle_distance * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos

@wp.kernel
def get_neighbor(
    grid: wp.uint64,
    smoothing_length: float,
    pt_array: wp.array(dtype=wp.vec3),
    nei_count: wp.array(dtype=wp.int32),
) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = pt_array[i]
    nei = wp.hash_grid_query(grid, x, smoothing_length)
    # count = 0
    nei_count[i] = 0
    for index in nei :
        # nei_x = pt_array[index]
        nei_count[i] += 1
        # print(index)
    # nei_count[i] = count

@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    size: float,
    damping_coef: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]


    if x[0] < -size:
        x = wp.vec3(-size, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > size:
        x = wp.vec3(size, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[1] > size :
        x = wp.vec3(x[0], size, x[2])
        v = wp.vec3(v[0], v[1] *damping_coef, v[2])

    # clamp y bot
    if x[1] < -size:
        x = wp.vec3(x[0], -size, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > size * 2.0:
        x = wp.vec3(x[0], x[1], size)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v
