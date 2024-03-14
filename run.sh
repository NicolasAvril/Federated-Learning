echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 9); do
    echo "Starting client $i"
    if [ $i -ge 9 ]; then
        echo "malveillant"
        python client_mal.py --node_id $i --attack_type label_flipping &
    else
        python client.py --node_id $i &
    fi
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait