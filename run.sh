##example full workflow
python doodler.py -c config_madeira/config_water.json
python doodler.py -c config_madeira/config_veg.json
python doodler.py -c config_madeira/config_anthro.json
python doodler.py -c config_madeira/config_substrate.json
python merge.py -c config_madeira/config_merge.json

