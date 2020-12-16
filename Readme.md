# Program synthesis transformer

Preparing data set command:
```
python clear.py --dataset <path-to-dataset-file> --output <output-dir-path>
```

Run model training:
```
python train.py
```
During the training stats are available on localhost link displayed in console.
Run tensorboard:
```
tensorboard --logdir=logs-<train start date  format: dd-MM-yyy>
```
Default link is [localhost:6006](http://localhost:6006/)

Interpreter from [link](https://github.com/nearai/program_synthesis).