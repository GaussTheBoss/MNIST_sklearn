# MNIST_sklearn
A sample model to demonstrate multi-class classification using **scikit-learn**. The saved model was trained on the MNIST dataset.

## Running Locally

To run this model locally, create a new Python 3.8.3 virtual environment
(such as with `pyenv`). Then, use the following command to update `pip`
and `setuptools`:

```
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `mnist.py`. To test all code at-once, run

```
python3 mnist.py
```

## Assets
- `./binaries/sklearn_mnist.pkl` is the trained sklearn.svm.SVC model.
- `input_schema.avsc` and `output_schema.avsc` are AVRO-compliant json files that detail the input and output schema, respectively.

## Scoring (Inference) Requests

### Sample Inputs

Choose `./data/sample_input.json` as the input file.

### Sample Output

`./data/sample_output.json` contains the output corrsesding to the input file above. Each line/output record is a dictionary with 2 keys: `predicted_probs` and `score`. Here's an example:
```json
{
    "predicted_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "score": 7
}
```