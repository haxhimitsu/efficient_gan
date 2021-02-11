#!/bin/sh
#python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase/01/ --mode train
#python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase/01/ --mode test --result_dir ./log/01/
#python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase/02/ --mode test --result_dir ./log/02/
#python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase/03/ --mode test --result_dir ./log/03/
#python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase/04/ --mode test --result_dir ./log/04/


python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase_fft/01/ --mode train  --log_dir ./log_fft01/
python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase_fft/01/ --mode test --result_dir ./log_fft01/01/
python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase_fft/02/ --mode test --result_dir ./log_fft01/02/
python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase_fft/03/ --mode test --result_dir ./log_fft01/03/
python3 efficient_GAN_torch.py  --dataset_path ~/Desktop/nagase_fft/04/ --mode test --result_dir ./log_fft01/04/