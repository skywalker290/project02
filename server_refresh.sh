cd ~/project02/Server02/
git pull

PID=$(lsof -t -i:5003)

# If the PID exists, kill the process
if [ -n "$PID" ]; then
    echo "Stopping Gunicorn server with PID: $PID"
    kill -9 $PID
    echo "Gunicorn server stopped"
else
    echo "No Gunicorn process found on port 5003"
fi

# Restart the Gunicorn server
echo "Starting Gunicorn server..."

echo "============================================" >> output.log
echo "Gunicorn server started at: $(date '+%Y-%m-%d %H:%M:%S')" >> output.log
echo "============================================" >> output.log
export PYTHONUNBUFFERED=1
nohup /home/drivool/.pyenv/versions/3.10.13/envs/yolo/bin/gunicorn -b 0.0.0.0:5003 app:app >> output.log 2>&1 &

echo "Gunicorn server started"