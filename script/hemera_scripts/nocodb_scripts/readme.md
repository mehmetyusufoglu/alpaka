# Scripts used to add data in json format to the previously created nocodb table
addBenchmarkResultJsonToNocoDBTable.sh is the only file called by the user.

# The python files here and the env.sh are from https://codebase.helmholtz.cloud/casus-datalad/datalad_automation.git

# command to send data in the json format to the already prepared nocodb table 
python3 ./enter_dataset.py -t TableID filename.json
# Example 
python3 ./enter_dataset.py -t ma3j18fd4jwxnls babelstream-gpu-cuda-nvcc-2025-03-05_15-03-1793a076.json

#For additional information please check:
https://codebase.helmholtz.cloud/casus-datalad/datalad_automation.git
https://github.com/nocodb/nocodb




