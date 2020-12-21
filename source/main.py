import numpy as np
import matplotlib.pylab as plt

AVOGADRO = 6.02214076e23 #avogadro constant
BOLTZMANN = 1.3806452e-23 #boltzmann constant

def wall_hit_check(position, velocity, box):
    ndimension = len(box)

    for i in range(ndimension):
        velocity[(position[:,i] <= box[i][0]) | (position[:,i] <= box[i][1])] *= -1

def integrate(position, velocity, forces, mass, dt):
    position += velocity * dt
    velocity +=forces * dt / mass[np.newaxis].T


def compute_force(mass, velocity, temp, relax, dt):
    natoms, ndimension = velocity.shape
    sigma = np.sqrt(2.0 * mass * temp * BOLTZMANN / (relax * dt))
    noise = np.random.randn(natoms, ndimension) + sigma[np.newaxis].T
    force = velocity * mass[np.newaxis].T / relax + noise

    return force


def run(**args):
    natoms, box, dt, temp = args["natom"], args["box"], args["dt"], args["temp"]
    mass, relax, nsteps = args["mass"], args["relax"], args["steps"]

    mass = np.ones(natoms) * mass / AVOGADRO
    ndimension = len(box)

    position = np.random.rand(natoms, ndimension)
    velocity = np.random.rand(natoms, ndimension)

    for i in range(ndimension):
        position[:,i] = box[i][0] + (box[i][1] - box[i][0] * position[:,i])

    step = 0
    output = []

    while step <= nsteps:
        step += 1
        forces = compute_force(mass, velocity, temp, relax, dt)
        integrate(position, velocity, forces, mass, dt)
        wall_hit_check(position, velocity, box)
        inst_temp = np.sum(np.dot(mass, (velocity - velocity.mean(axis=0))**2) / (BOLTZMANN * 3 * natoms))

        output.append([dt * step, inst_temp])

    return np.array(output)


if __name__ == "__main__":
    params = {
        "natom": 1000,
        "radius": 120e-12,
        "mass": 1e-3,
        "dt": 1e-15,
        "relax": 1e-13,
        "temp": 300,
        "steps": 10000,
        "freq": 100,
        "box": ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
        "ofname": "traj-hydrogen.dump"
    }

    output = run(**params)
    plt.plot(output[:,0] * 1e12, output[:,1])
