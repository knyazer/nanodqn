# runpod is not a standard executable - you can find it at https://github.com/knyazer/config

runpod_start:
  runpod start --bid 0.42 --gpuCount 2 # set to your default config

runpod_setup:
  runpod ssh "apt-get update && apt-get install -y rsync screen && git clone https://github.com/knyazer/nanodqn" || true
  runpod upload results nanodqn/ -s
  runpod ssh "cd nanodqn && ./run.sh"

runpod_download:
  runpod download nanodqn/results . -s
