# Stock movement prediction

The portfolio described one log-return feature and a neural network with 16 ReLU hidden units and 3 Softmax outputs. This reconstruction labels the next day's closing-price movement as down, flat, or up. Flat means exactly unchanged, so that class may be rare in real price series.

Create a CSV with `Date,Close` columns sorted by the loader in chronological order. The file is deliberately not supplied: the original `Apple.csv` and training results were not attached.

```bash
python -m pip install -e '.[stock]'
python -m stem_projects.train_stock data/private/prices.csv --epochs 20 --output outputs/stock
```

The command trains on the first 80% of feature/label pairs and reports classification accuracy and a confusion matrix for the last 20%. It standardizes the one feature using training observations only and saves `model.keras` and `metrics.json`. You can change CSV column names with `--date-column` and `--close-column`.

The chronological holdout and training-only scaling make the recreated experiment easier to evaluate on new data. The portfolio shows the model architecture and training setup but does not state a validated accuracy figure. A three-class classifier trained on historical prices is not evidence of a profitable strategy.
