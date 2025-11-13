# Machine Learning Classification

This collection of scripts supports the training of ML models,
inference of those models and porting the trained models into
p0f by transpiling the model to C code.

- `train.py` - Training ML Models
- `predict.py` - Inference of ML Models
- `port.py` - Porting ML Models into p0f

### Training ML Models

First things first, we have to train the model and produce a
.joblib file that contains the model's Pipeline object.

We can do this by using the first script on the list, `train.py`.
We assume a .csv file exists in the data/ directory, for example
`data/small_all.csv`, which contains a small amount of training
data, so not the full dataset. The command thus becomes the
following.

```sh
uv run train.py --file data/small_all.csv
```

This produces a .joblib file in the current working directory.
This file can then be used for inference, in the next step.

### Inference of ML Models

Using the second script on the list, `predict.py`, we can query
for inference. This script has two important flags to take
into account.

- `--file` - uses a file for inference.
- `--string` - uses a string for inference.

In the next example, we will use a file for inference.

```sh
uv run predict.py --file data/7015-small-no-id.csv
```

This produces an output where we should see all 7015 being
returned in stdout.

### Porting ML Models into p0f

Finally, we can use the third script, `port.py`, to transpile
the specified trained model into C code and write it into
p0f as `classifier.c`.

The command has one important flag, and important positional
arguments.

- `--file` - The input .joblib file.
- `$1, ..., $n` - The positional arguments representing the
                identifiers of the os in human readable form
                in the order of the virtual machine ID: 7011,
                7012, 7013, ...

```sh
uv run port.py --file models/best_model_DecisionTree.joblib "unix Arch-Linux" "unix openSUSE-Leap-16.0-Linux" "unix Fedora-43.1.6-Linux" "unix Fedora-42.1.1-Linux" "unix Debian-12.5.0- Linux" "win Windows-Server-2022" "win Windows-Server-2025"
```

That's it! You can now rebuild p0f by running `./src/build.sh`
in the project root. The compiled executable will be found
in the src/ directory as p0f, and can be run by running
`./src/p0f`
