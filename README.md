# Yo this is the repo for the CSC 4770 Term Project


#### In this repo will be the code for our Project. 

How to use:

docker compose up --build when you need ot create the containers for the first time
docker compose up when they have already been built

How to send a job manually:
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

To start the server and start the client (which creates the matrices automatically) run these commands:
docker compose up, this starts the docker containers and runs the app.py.

then you can run this command (either in a separate terminal or the same):
python3 client.py

that will automatically create the matrix that you specified in the client.py file and distrubute it across the nodes

How to veiw the results, paste the id of that request after result/
curl http://localhost:5000/result/287d5acf-98fd-48ee-af31-968fe11f9fd6