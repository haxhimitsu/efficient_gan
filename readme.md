# Efficient gan
Efficient gan pytorch implementation
# Usage
# Train and Validation
```bash
python3 efficient_GAN_torch.py  --dataset_path ./dataset01/ --mode train  --log_dir ./log_fft02/  --result_dir ./log_fft02/ --max_epochs 60
python3 efficient_GAN_torch.py  --dataset_path ./dataset/01/ --mode test --result_dir ./log_fft02/01/ --log_dir ./log_fft02/
python3 efficient_GAN_torch.py  --dataset_path ./dataset/02/ --mode test --result_dir ./log_fft02/02/ --log_dir ./log_fft02/
python3 efficient_GAN_torch.py  --dataset_path ./dataset/03/ --mode test --result_dir ./log_fft02/03/ --log_dir ./log_fft02/
python3 efficient_GAN_torch.py  --dataset_path ./dataset/04/ --mode test --result_dir ./log_fft02/04/ --log_dir ./log_fft02/
```
# visualize anomaly score
```
python3 make_graph.py --csv_path {your csv file name}
```
# argments specification
```ptyhon
parser.add_argument("--dataset_path",required=True,help="path to root dataset directory")
parser.add_argument("--max_epochs", type =int ,default=80,help="set trim width")
parser.add_argument("--save_weight_name", type=str,default="test",help="set trim height")
parser.add_argument("--batch_size", default=64, type=int,help="output path")
parser.add_argument("--log_dir",  default="./log/",help="log_path")
parser.add_argument("--mode",  type=str,required=True,default="train",help="log_path")
parser.add_argument("--result_dir", type=str,default="./img_result/",help="log_path")
```

# Dataset structure
```python
--dataset01
    --train
        --class3
    --val
        --class1
--dataset02
    --train
        --class3
    --val
        --class2
--dataset03
    --train
        --class3
    --val
        --class3
--dataset04
    --train
        --class3
    --val
        --class4
```