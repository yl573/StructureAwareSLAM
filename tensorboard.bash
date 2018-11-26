tensorboard --logdir ../logs --host 0.0.0.0 --port 6006 &

../ngrok http 6006 &

curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"