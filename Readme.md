# Program synthesis transformer

## Dataset

Firs you need to download Algolisp Dataset. 
There are three files to download:
- Train: https://www.dropbox.com/s/qhun6kml9yb2ui9/metaset3.train.jsonl.gz
- Dev: https://www.dropbox.com/s/aajkw83j2ps8bzx/metaset3.dev.jsonl.gz
- Test: https://www.dropbox.com/s/f1x9ybkjpf371cp/metaset3.test.jsonl.gz

The original source is [repository](https://github.com/nearai/program_synthesis/tree/master/program_synthesis/algolisp).

After downloading and unpacking data set you need to clear inconsistent data from it.
Run commands below:
- Preparing dataset with output dir "cleared_data":
```
python clear.py --dataset <path-to-dataset-file> --output <output-dir-path (default "cleared_data")>
```

- Filtering dataset:
```
python filter_programs.py --dataset <path-to-dataset-file> --output <output-dir-path (default "filtered_data")>
```

## Training

Run model training:
```
python train.py
```

Run tensorboard to see training statistics:
```
tensorboard --logdir=logs-<train start date  format: dd-MM-yyy>
```
Default link is [localhost:6006](http://localhost:6006/)

See model summary:
```
python run.py --summary
```

## Testing

Coming soon.

Interpreter from [link](https://github.com/nearai/program_synthesis).