# Source
https://www.youtube.com/watch?v=bbAWNb1j1w0

# Prerequisites 
Add path to sumo
```
export SUMO_HOME=".../Program Files (x86)/Eclipse/Sumo"
```

# How to run
## Normal simulation
1. python main.py
2. Click start simulation in opened window
3. Click Enter in terminal
4. Look as simulation runs

## AC training
1. python main-ac-train.py
2. Click start simulation in opened window

## AC test
1. python main-ac-test.py
2. Click start simulation in opened window

## Visualization
1. jupyter notebook
2. In browser open `visualize.ipynb`

# Files
* `ac-test.csv` - Time series of ac test simulation
* `ac-train.csv` - Time series of ac train simulation
* `normal.csv` - Time series of default SUMO simulation
* `Traci.net.cml` - Grid layout
* `Traci.netecfg` / `Traci.sumocfg` - SUMO project setup
* `Traci.rou.xml` - Demand description. Start and end of cars
* `

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

