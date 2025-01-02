cd ~/project02/Server02/
git pull

PID=$(lsof -t -i:5000)

# If the PID exists, kill the process
if [ -n "$PID" ]; then
    echo "Stopping Gunicorn server with PID: $PID"
    kill -9 $PID
    echo "Gunicorn server stopped"
else
    echo "No Gunicorn process found on port 5000"
fi

# Restart the Gunicorn server
echo "Starting Gunicorn server..."

echo "============================================" >> output.log
echo "Gunicorn server started at: $(date '+%Y-%m-%d %H:%M:%S')" >> output.log
echo "============================================" >> output.log
nohup gunicorn -b localhost:5000 app:app >> output.log 2>&1 &

echo "Gunicorn server started"