import libsumo as traci
import pandas as pd
import matplotlib.pyplot as plt

Sumo_config = [
    'sumo-gui.exe',
    '-c', 'Traci.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1'
]

traci.start(Sumo_config)

vehicle_speed = 0
total_speed = 0

# input("Have you started simulation?")

data = []


def analysis():
    wait = 0
    queue_len = 0
    for v in traci.vehicle.getIDList():
        wait += traci.vehicle.getWaitingTime(v)
    for e in traci.edge.getIDList():
        queue_len += traci.edge.getLastStepHaltingNumber(e)
    data.append({
        'time': traci.simulation.getTime(),
        'queue_len': queue_len,
        'wait': wait
    })


while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    analysis()


traci.close()

df = pd.DataFrame(data)
df.to_csv("normal.csv")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(df.time, df.wait)
ax1.set_ylabel('Cummulative wait time')

ax2.plot(df.time, df.queue_len)
ax2.set_ylabel('Cummulative queue length')

plt.tight_layout()
plt.show()
