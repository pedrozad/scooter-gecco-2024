# GECCO-2024 scooter

## To ejecute the project
```bash
pip install -r requirements.txt

python multiobjective_main_arguments.py -seed [SEED] -pc [PROB_CROSS] -pm [PROB_FLIP] -POB [POPULTATION] -CPUS [CPUS] -a [ALGORITHM] -f [over_all_base_withBus.csv]
```

Python version 3.10.12 64 bits
## Structure of the project
Main
 * multiobjective_main_arguments.py
 * data-osm
    * Malaga-Subway
        * districts-Malaga-Subway-data-with-nodes.csv
        * map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY-wBUS-wMetro.gpkg
        * pair_less_than_3600_new_points.csv
 * over_all_base_withBus.csv 

## Data

The data folder are in https://uma365-my.sharepoint.com/:f:/g/personal/pedroza_uma_es/Er7uXn2sKLxPvaqNpDpj6hoB0Muxh8Bp1NedMk7yQUoTZw?e=VvfMCr

## Arguments
Example of arguments can be found in arguments.txt
