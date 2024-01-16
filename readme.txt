Deep Q Learning Agent Notes:

1) Thx to the Medium post: https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472#c87e

For creating such good content and making me curious about it.

2) Thanks to the youtube channel: sentdex 

For the playlist about Deep Q Learning and for being so good explaining what you do.

3) For deploying the container:

docker build -t dql_test --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile_gpu .

docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    dql_test \
    bash

