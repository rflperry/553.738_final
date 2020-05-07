# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

# Total population, N.
N = 1
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 0.1, 0.1
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
# zeta = exponential decay rate for immunity
beta, gamma, zeta = 0.5, 0.1, 0.01
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, zeta):
    S, I, R = y
    dSdt = -beta * S * I / N + zeta * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - zeta * R
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')


ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
    
plt.savefig('test.png')


plt.clf()


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

# I0_range = np.logspace(-2, 0, 50)
I0_range = np.linspace(0.02, 0.3, 20)

for I0 in tqdm(I0_range):
    y0 = N-I0 - R0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, zeta))
    S, I, R = ret.T

    ax.plot(S, I, c='g')

plt.xlabel('Susceptible')
plt.ylabel('Infected')
# plt.legend([f'$I_0 = {i}$' for i in I0_range])
plt.savefig('phaseportrait_S_I.png')




plt.clf()
# # Load case data

# import csv

# reader = csv.reader(open('cases.csv','r'))
# total_cases = {}

# c = 0

# for line in reader:
#     c+=1
#     if c == 1:
#         continue
#     if line[0] not in total_cases:
#         total_cases[line[0]] = 0
#     total_cases[line[0]] += int(line[3])

#     c+=1

# plt.plot(np.arange(len(total_cases)), list(total_cases.values()))
# plt.savefig('total_cases.png')