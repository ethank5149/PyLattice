import numpy as np
import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import  dynamicsymbols, LagrangesMethod
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from functools import partial
import dill
from tqdm import trange

rng = np.random.default_rng()
dill.settings['recurse'] = True

DEFAULT_PARAMS = {
    'l0' : 1. / 2.,
    'm' : 1.,
    'g' : 0,  # 9.80665,
    'k' : 1
}
RAND_PARAMS = {
    # 'v' : 0.1,
    'v' : np.sqrt(0.5 * (2 - np.sqrt(2))**2 * DEFAULT_PARAMS['k'] * DEFAULT_PARAMS['l0'] ** 2 / DEFAULT_PARAMS['m']),
}
DEFAULT_ICS = np.asarray([
    -DEFAULT_PARAMS['l0'], 0, DEFAULT_PARAMS['l0'], 
    -DEFAULT_PARAMS['l0'], 0, DEFAULT_PARAMS['l0'],
    -DEFAULT_PARAMS['l0'], 0, DEFAULT_PARAMS['l0'],

     DEFAULT_PARAMS['l0'],  DEFAULT_PARAMS['l0'],  DEFAULT_PARAMS['l0'], 
                        0,                     0,                     0,
    -DEFAULT_PARAMS['l0'], -DEFAULT_PARAMS['l0'], -DEFAULT_PARAMS['l0'],

    -RAND_PARAMS['v'] / np.sqrt(2), 0, RAND_PARAMS['v'] / np.sqrt(2), 
                 -RAND_PARAMS['v'], 0,              RAND_PARAMS['v'], 
    -RAND_PARAMS['v'] / np.sqrt(2), 0, RAND_PARAMS['v'] / np.sqrt(2), 

     RAND_PARAMS['v'] / np.sqrt(2),  RAND_PARAMS['v'],  RAND_PARAMS['v'] / np.sqrt(2),
                                 0,                 0,                              0,
    -RAND_PARAMS['v'] / np.sqrt(2), -RAND_PARAMS['v'], -RAND_PARAMS['v'] / np.sqrt(2),
    ])


def progress_bar(frame, total):
    iteration = frame + 1
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(100 * iteration // total)
    progress = '█' * filled_length + '-' * (100 - filled_length)
    print(f'\rSaving Animation: |{progress}| {percent}% ', end="\r")
    # if iteration == total:
    if frame == total:
        print()

def gen_rand_ics(
        l0=DEFAULT_PARAMS['l0'], 
        v=RAND_PARAMS['v']
        ):
    rand_v = rng.uniform(high=v, size=9) 
    theta = rng.uniform(high=2 * np.pi)
    return np.asarray([
        -l0, 0, l0, 
        -l0, 0, l0,
        -l0, 0, l0,
        
         l0,  l0,  l0, 
          0,   0,   0,
        -l0, -l0, -l0,

        -rand_v[0] / np.sqrt(2),                         0, rand_v[2] / np.sqrt(2), 
                     -rand_v[3], rand_v[4] * np.cos(theta),              rand_v[5], 
        -rand_v[6] / np.sqrt(2),                         0, rand_v[8] / np.sqrt(2), 
        
         rand_v[0] / np.sqrt(2),                 rand_v[1],  rand_v[2] / np.sqrt(2),
                              0, rand_v[4] * np.sin(theta),                       0,
        -rand_v[6] / np.sqrt(2),                -rand_v[7], -rand_v[8] / np.sqrt(2),
    ])

DEFAULT_ICS = gen_rand_ics()

def solve_symbolic(
        verbose=True
        ):
        if verbose: print('Solving symbolic problem... ', end='', flush=True)
        t, l0, k, m, g = sp.symbols('t l0 k m g')
        q = dynamicsymbols(" ".join(f"q_{_}" for _ in range(18)))
        p = dynamicsymbols(" ".join(f"q_{_}" for _ in range(18)), 1)
        a = dynamicsymbols(" ".join(f"q_{_}" for _ in range(18)), 2)
        x = [q[0],  q[1],  q[2],  q[3],  q[4],  q[5],  q[6],  q[7],  q[8]]
        y = [q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17]]
        dx = [p[0],  p[1],  p[2],  p[3],  p[4],  p[5],  p[6],  p[7],  p[8]]
        dy = [p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17]]
        T = (m / 2) * (
            dx[0] ** 2 + dy[0] ** 2 + 
            dx[1] ** 2 + dy[1] ** 2 + 
            dx[2] ** 2 + dy[2] ** 2 + 
            dx[3] ** 2 + dy[3] ** 2 + 
            dx[4] ** 2 + dy[4] ** 2 + 
            dx[5] ** 2 + dy[5] ** 2 + 
            dx[6] ** 2 + dy[6] ** 2 + 
            dx[7] ** 2 + dy[7] ** 2 + 
            dx[8] ** 2 + dy[8] ** 2
            )
        s_sqr = lambda x, y, l0 : (sp.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) - l0) ** 2
        V = (k / 2) * (  # Spring Potential
            s_sqr([x[0], x[1]], [y[0], y[1]], l0) + 
            s_sqr([x[1], x[2]], [y[1], y[2]], l0) + 
            s_sqr([x[3], x[4]], [y[3], y[4]], l0) + 
            s_sqr([x[4], x[5]], [y[4], y[5]], l0) + 
            s_sqr([x[6], x[7]], [y[6], y[7]], l0) + 
            s_sqr([x[7], x[8]], [y[7], y[8]], l0) + 
            s_sqr([x[0], x[3]], [y[0], y[3]], l0) + 
            s_sqr([x[1], x[4]], [y[1], y[4]], l0) + 
            s_sqr([x[2], x[5]], [y[2], y[5]], l0) + 
            s_sqr([x[3], x[6]], [y[3], y[6]], l0) + 
            s_sqr([x[4], x[7]], [y[4], y[7]], l0) + 
            s_sqr([x[5], x[8]], [y[5], y[8]], l0) + 
            s_sqr([x[0], x[4]], [y[0], y[4]], l0) + 
            s_sqr([x[1], x[5]], [y[1], y[5]], l0) + 
            s_sqr([x[3], x[7]], [y[3], y[7]], l0) + 
            s_sqr([x[4], x[8]], [y[4], y[8]], l0) + 
            s_sqr([x[1], x[3]], [y[1], y[3]], l0) + 
            s_sqr([x[2], x[4]], [y[2], y[4]], l0) + 
            s_sqr([x[4], x[6]], [y[4], y[6]], l0) + 
            s_sqr([x[5], x[7]], [y[5], y[7]], l0)
            ) + \
            m * g * (  # Gravitational Potential
                y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7] + y[8]
                )
        L = T - V
        Kinetic, Potential, H = T, V, T + V
        LM = LagrangesMethod(L, q)
        EL = LM.form_lagranges_equations()
        EOM = sp.solve(EL, a)
        EOMsoln = Matrix([EOM[a[_]] for _ in range(len(a))])
        if verbose: print('Done!')
        if verbose: print('Caching solution... ', end='', flush=True)
        ODEsystem = sp.utilities.lambdify([t, [*q, *p], k, l0, m, g], [*p, *EOMsoln])
        Hamiltonian = sp.utilities.lambdify([t, [*q, *p], k, l0, m, g], H)
        Kinetic = sp.utilities.lambdify([t, [*q, *p], k, l0, m, g], Kinetic)
        Potential = sp.utilities.lambdify([t, [*q, *p], k, l0, m, g], Potential)
        dill.dump(ODEsystem, open("./cache/pylattice_cached_soln", "wb"))
        dill.dump(Hamiltonian, open("./cache/pylattice_cached_h", "wb"))
        dill.dump(Kinetic, open("./cache/pylattice_cached_kinetic", "wb"))
        dill.dump(Potential, open("./cache/pylattice_cached_potential", "wb"))
        if verbose: print('Done!')
        return ODEsystem, Hamiltonian, Kinetic, Potential

def solve_numeric(
        tf, 
        fps, 
        ics, 
        params, 
        verbose=True
        ):
    if verbose: print('Checking for cached solutions... ', end='', flush=True)
    try:
        ODEsystem = dill.load(open("./cache/pylattice_cached_soln", "rb"))
        Hamiltonian = dill.load(open("./cache/pylattice_cached_h", "rb"))
        Kinetic = dill.load(open("./cache/pylattice_cached_kinetic", "rb"))
        Potential = dill.load(open("./cache/pylattice_cached_potential", "rb"))
        if verbose: print('Done! (Solutions found)')
    except:
        if verbose: print('Done! (One or more missing)')
        ODEsystem, Hamiltonian, Kinetic, Potential = solve_symbolic(verbose=verbose)

    if verbose: print('Solving numerical problem... ', end='', flush=True)
    frames = tf * fps
    t_eval = np.linspace(0, tf, frames)

    ode = partial(ODEsystem, **params)
    sol = solve_ivp(ode, [0, tf], ics, t_eval=t_eval)

    x, y, dx, dy = np.split(sol.y, 4)
    l0, k, m, g = params['l0'], params['k'], params['m'], params['g']

    energy = Hamiltonian(t_eval, sol.y, k, l0, m, g)
    kinetic = Kinetic(t_eval, sol.y, k, l0, m, g)
    potential = Potential(t_eval, sol.y, k, l0, m, g)
    if verbose: print('Done!')
    return [x, y, dx, dy], [energy, kinetic, potential]

def simulate(
        tf, 
        fps, 
        ics=DEFAULT_ICS, 
        params=DEFAULT_PARAMS, 
        run=0,
        verbose=True
):
    [x, y, dx, dy], [energy, kinetic, potential] = solve_numeric(tf=tf, fps=fps, ics=ics, params=params, verbose=verbose)
    energy_loss_percent = 100 * (energy - energy[0]) / energy[0]
    kinetic_unitless = kinetic / kinetic[0]
    potential_unitless = potential / potential[0]

    if verbose: print('Creating animation... ', end='', flush=True)
    frames = tf * fps
    dt = tf / frames
    t_eval = np.linspace(0, tf, frames)

    ##
    energy_loss_fit = linregress(t_eval, energy_loss_percent)
    energy_loss_fit_plot = energy_loss_fit.slope * t_eval + energy_loss_fit.intercept
    ##

    fig = plt.figure(layout="constrained", figsize=(19.2, 10.80))
    fig.suptitle('PyLattice-N9\nWritten by: Ethan Knox')
    gs = GridSpec(2, 2, figure=fig)
    ax3 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax3)
    plt.setp(ax2.get_xticklabels(), visible=False)  # Replicate subplots sharex
    ax1 = fig.add_subplot(gs[:, 0])

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    ax1.set_xlim((1.15 * min_x, 1.15 * max_x))
    ax1.set_ylim((1.15 * min_y, 1.15 * max_y))        

    ax1.set_aspect('equal')
    ax1.set_xlabel(r'X [m]')
    ax1.set_ylabel(r'Y [m]')
    ax2.set_ylabel(r'Energy Loss [%]')
    ax3.set_xlabel(r'$t$ $[s]$')
    ax3.set_ylabel(r'Energy $[E]/[E_0]$')

    ax3.axhline(y=1, xmin=t_eval[0], xmax=t_eval[-1], linestyle='-', color='black')
    ax3.plot(t_eval, kinetic_unitless, '-', lw=1.5, color='red', label='Kinetic Energy')
    ax3.plot(t_eval, potential_unitless, '-', lw=1.5, color='blue', label='Potential Energy')
    kinetic_plot, = ax3.plot([], [], 'o', lw=3, color='red')
    potential_plot, = ax3.plot([], [], 'o', lw=3, color='blue')
    ax3.legend()
    ax3.grid()

    ax2.axhline(y=0, xmin=t_eval[0], xmax=t_eval[-1], linestyle='-', color='black')
    ax2.plot(t_eval, energy_loss_percent, '-', lw=1.5, color='purple', label='Total Energy')
    ax2.plot(t_eval, energy_loss_fit_plot, linestyle='--', color='black', label=rf'$r^2={energy_loss_fit.rvalue:.4f},$ $p={energy_loss_fit.pvalue:.4f}$')
    energy_loss_plot, = ax2.plot([], [], 'o', lw=3, color='purple')
    ax2.legend()
    ax2.grid()

    spring01, = ax1.plot([], [], '--', lw=2, color='black')
    spring02, = ax1.plot([], [], '--', lw=2, color='black')
    spring03, = ax1.plot([], [], '--', lw=2, color='black')
    spring04, = ax1.plot([], [], '--', lw=2, color='black')
    spring05, = ax1.plot([], [], '--', lw=2, color='black')
    spring06, = ax1.plot([], [], '--', lw=2, color='black')
    spring07, = ax1.plot([], [], '--', lw=2, color='black')
    spring08, = ax1.plot([], [], '--', lw=2, color='black')
    spring09, = ax1.plot([], [], '--', lw=2, color='black')
    spring10, = ax1.plot([], [], '--', lw=2, color='black')
    spring11, = ax1.plot([], [], '--', lw=2, color='black')
    spring12, = ax1.plot([], [], '--', lw=2, color='black')
    spring13, = ax1.plot([], [], '--', lw=2, color='black')
    spring14, = ax1.plot([], [], '--', lw=2, color='black')
    spring15, = ax1.plot([], [], '--', lw=2, color='black')
    spring16, = ax1.plot([], [], '--', lw=2, color='black')
    spring17, = ax1.plot([], [], '--', lw=2, color='black')
    spring18, = ax1.plot([], [], '--', lw=2, color='black')
    spring19, = ax1.plot([], [], '--', lw=2, color='black')
    spring20, = ax1.plot([], [], '--', lw=2, color='black')

    mass1, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass2, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass3, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass4, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass5, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass6, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass7, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass8, = ax1.plot([], [], 'o', lw=6, color='blue')
    mass9, = ax1.plot([], [], 'o', lw=6, color='blue')

    def animate(i):
        mass1.set_data([x[0][i]], [y[0][i]])
        mass2.set_data([x[1][i]], [y[1][i]])
        mass3.set_data([x[2][i]], [y[2][i]])
        mass4.set_data([x[3][i]], [y[3][i]])
        mass5.set_data([x[4][i]], [y[4][i]])
        mass6.set_data([x[5][i]], [y[5][i]])
        mass7.set_data([x[6][i]], [y[6][i]])
        mass8.set_data([x[7][i]], [y[7][i]])
        mass9.set_data([x[8][i]], [y[8][i]])

        spring01.set_data([x[0][i], x[1][i]], [y[0][i], y[1][i]])
        spring02.set_data([x[1][i], x[2][i]], [y[1][i], y[2][i]])
        spring03.set_data([x[3][i], x[4][i]], [y[3][i], y[4][i]])
        spring04.set_data([x[4][i], x[5][i]], [y[4][i], y[5][i]])
        spring05.set_data([x[6][i], x[7][i]], [y[6][i], y[7][i]])
        spring06.set_data([x[7][i], x[8][i]], [y[7][i], y[8][i]])

        spring07.set_data([x[0][i], x[3][i]], [y[0][i], y[3][i]])
        spring08.set_data([x[1][i], x[4][i]], [y[1][i], y[4][i]])
        spring09.set_data([x[2][i], x[5][i]], [y[2][i], y[5][i]])
        spring10.set_data([x[3][i], x[6][i]], [y[3][i], y[6][i]])
        spring11.set_data([x[4][i], x[7][i]], [y[4][i], y[7][i]])
        spring12.set_data([x[5][i], x[8][i]], [y[5][i], y[8][i]])

        spring13.set_data([x[0][i], x[4][i]], [y[0][i], y[4][i]])
        spring14.set_data([x[1][i], x[5][i]], [y[1][i], y[5][i]])
        spring15.set_data([x[3][i], x[7][i]], [y[3][i], y[7][i]])
        spring16.set_data([x[4][i], x[8][i]], [y[4][i], y[8][i]])

        spring17.set_data([x[1][i], x[3][i]], [y[1][i], y[3][i]])
        spring18.set_data([x[2][i], x[4][i]], [y[2][i], y[4][i]])
        spring19.set_data([x[4][i], x[6][i]], [y[4][i], y[6][i]])
        spring20.set_data([x[5][i], x[7][i]], [y[5][i], y[7][i]])

        energy_loss_plot.set_data([t_eval[i]], [energy_loss_percent[i]])
        kinetic_plot.set_data([t_eval[i]], [kinetic_unitless[i]])
        potential_plot.set_data([t_eval[i]], [potential_unitless[i]])

        return (
            mass1, mass2, mass3, 
            mass4, mass5, mass6, 
            mass7, mass8, mass9, 
            spring01, spring02, spring03, spring04, spring05, 
            spring06, spring07, spring08, spring09, spring10, 
            spring11, spring12, spring13, spring14, spring15, 
            spring16, spring17, spring18, spring19, spring20, 
            energy_loss_plot, kinetic_plot, potential_plot,
        )


    anim = animation.FuncAnimation(fig, animate, len(t_eval), interval=dt * 1000)
    if verbose: print('Done!')
    ics_tag = 'ics=[' + ','.join((f'{ic:.6f}' for ic in ics)) + ']'
    anim.save(
        f'./resources/pylattice_n9_{str(run).zfill(2)}.mp4', 
        progress_callback = progress_bar,
        metadata=dict(
            title='PyLattice',
            artist='Ethan Knox',
            comment=ics_tag
            )
        )
    plt.close()
    if verbose: print('Simulation Complete!')
    return None

def main(
        tf=15, 
        fps=60, 
        ics=DEFAULT_ICS, 
        params=DEFAULT_PARAMS,
        run='ex',
        verbose=True
        ):
    simulate(tf=tf, fps=fps, ics=ics, params=params, run=run, verbose=verbose)
    return None

def multi_run(runs):
    for run in trange(runs):
        rand_ics = gen_rand_ics()
        main(ics=rand_ics, run=run, verbose=False)
    return None


if __name__ == "__main__":
    multi_run(3)
    # main()
