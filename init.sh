pythonPath="/home/sysadmin/aitp/lib"
pythonPath+=":/home/sysadmin/aitp/lib/models"
pythonPath+=":/home/sysadmin/aitp/lib/services"
pythonPath+=":/home/sysadmin/aitp/lib/datasets"
pythonPath+=":/home/sysadmin/aitp/lib/var"

export PYTHONPATH=$pythonPath

# Make sure logs folder exists
mkdir -p /home/sysadmin/aitp/logs
mkdir -p /home/sysadmin/aitp/datasets
mkdir -p /home/sysadmin/aitp/models
