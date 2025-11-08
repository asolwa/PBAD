# Source
https://www.youtube.com/watch?v=bbAWNb1j1w0

# How to run
1. python main.py
2. Click start simulation in opened window
3. Click Enter in terminal
4. Look as simulation runs

# Demand
Demand was created using `randomTrips.py` script located in SUMO_HOME/tools
```
randomTrips.py -n Traci.net.xml -e 50
```
It generates 50 routes within the provided map. The resulting file should be
renamed to `Traci.rou.xmL so that it can be used by the simulation.

Additionally, it also creates a file named `trips.trips.xml`, though its
purpose is unclear.

## Source
* https://sumo.dlr.de/docs/Demand/Introduction_to_demand_modelling_in_SUMO.html

