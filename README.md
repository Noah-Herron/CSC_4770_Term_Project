# CSC 4770 Term Project: Distributed Matrix Multiplication

This repository contains the code for our distributed systems project. The goal is to calculate matrix multiplication in parallel by distributing the workload across multiple Docker containers.

## How It Works

* The system runs on a master-worker architecture defined in the `docker-compose.yaml` file.


* The master node runs a Flask API on port 5000 to receive user uploads and a TCP server on port 6000 to communicate with the workers.


* When you send matrices to the `/multiply` endpoint, `app.py` splits Matrix A into chunks.


* Currently, the chunk size is set to 2 rows to force multiple jobs for our early testing.


* Worker nodes connect to the master, ask for a job, compute the dot product using NumPy, and send the chunk back.


* The master node stores these results in an in-memory database, sorts them by the starting row, and reconstructs the final matrix once all jobs are completed.



## How to Run It

We are using Docker to handle the environment and networking. The master node will start first, and 3 worker replicas will automatically start and connect to it.

1. To create and start the containers for the first time, run: `docker compose up --build`.


2. If the images are already built and you just need to start them, run: `docker compose up`.


###### Note: If you change the code you have to rebuild.


## How to Test It

You have two ways to send matrices to the master node:

### 1. Using the Client Script (Recommended)

We have a `client.py` script that automatically generates matrices with random integers between 1 and 10.

* By default, it creates two 100x100 matrices.


* Once your Docker containers are running, open a new terminal and run: `python3 client.py`.


* The script will upload the payload, grab the request ID, and poll the server every 2 seconds until the final matrix is ready.



### 2. Manual cURL Command

If you want to test a specific small matrix, you can POST it manually.

```bash
curl -X POST http://localhost:5000/multiply -H "Content-Type: application/json" -d '{
    "matrix_A": [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    "matrix_B": [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ]
}'

```

The server will return a JSON response with a `request_id`. To view the final result, paste that ID at the end of the `/result/` endpoint:

```bash
curl http://localhost:5000/result/<your_request_id>

```

## How to Add Stuff

If you are jumping in to help write code, here is where everything lives:

* **Routing & Logic:** All the Flask endpoints, job queueing, worker computations, and aggregation logic are inside `app.py`.


* **Test Matrix Size:** If you want to benchmark larger matrices, open `client.py` and change the `matrix_size = 100` variable at the bottom.


* **Dependencies:** If your new code needs a new Python package, add it to `requirements.txt`. The `Dockerfile` reads this file to install packages when building the container images.