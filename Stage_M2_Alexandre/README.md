run training:
python run.py -t -b ./config/autoencoder/autoencoder_kl-f16.yaml --no-test -n "kl-f16" -p "test" --scale_lr

resume training:
python run.py -t -r "./logs/01-07-2024_135411_kl-f16" -b ./config/autoencoder/autoencoder_kl-f16.yaml --no-test -p "test" --scale_lr

tensorboard:
python -m tensorboard.main --logdir="./logs/01-07-2024_135411_kl_f16"