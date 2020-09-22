## Monte Carlo simulation program for generalized skyrmion models

### Description

A program for Monte Carlo simulations of the skyrmion model 
> * Skyrmion model : the model with DMI interactions between spins
> * Built-in MC algorithms: PT, annealings, single spin flips, over-relaxation
> * 1D, 2D, thin film (BC)
> * Lattice : triangular lattice, cubic lattice

### Requirements
#### CUDA >6.0
#### boost
#### nlohmann/json

***nlohmann_json/3.1.1*** is required. Plesase do not use the version >3.2 since there is conflict with nvcc.
The default method is using ***conan***.
```bash
cd src
conan install ..

```

### Installation
```bash
cd src
make clean
make
```

### Usage
***config.json*** is the configuration file for the simulation. One can adjust it to tune the simulation.
***config.json*** include a table of temperatures and fields.
Standard way is to generate config.json by tools/config.py

```bash
python tools/config.py params/params.json Tlists/Tlist Hlists/Hlist
```
This can generate ***config.json*** with grid-like temperature and field points for parallel tempering.

Run the simulation by
```bash
./bin/simulation_TRI
```

### Analysis

One can use ***analysis/MCdata.py*** to analysis the data easily. Please look up ***analysis/example.ipynb***.

### TODO list
> * The simulation for square lattice(cubic).

### Contributor

Po-Kuan Wu
