echo "Starting server"
python server.py &
sleep 3  # Give the server enough time to start

# Assume results.txt is where we append the accuracy results
echo "ClientsMalveillants,Précision" > results.txt

# Function to launch clients with a specific number of malicious ones
launch_clients() {
    local num_malicious=$1
    local total_clients=4
    local malicious_start=$((total_clients - num_malicious))

    # Launch honest clients
    for i in `seq 0 $((malicious_start - 1))`; do
        echo "Starting honest client $i"
        python client.py --node_id ${i} --n ${total_clients} &
    done

    # Launch malicious clients
    for i in `seq ${malicious_start} $((total_clients - 1))`; do
        echo "Starting malicious client $i"
        python client_mal.py --node_id ${i} --n ${total_clients} &
    done

    wait
    # Simulate collecting accuracy from the server's output or a log file
    local accuracy=0.85 # Placeholder for real accuracy retrieval mechanism
    echo "${num_malicious},${accuracy}" >> results.txt
}

# Run experiments with 1 to 3 malicious clients
for num_malicious in `seq 1 3`; do
    echo "Experiment with ${num_malicious} malicious clients"
    launch_clients ${num_malicious}
    sleep 5  # Adjust based on how long your training takes
done

# Remember to kill your server process if needed
