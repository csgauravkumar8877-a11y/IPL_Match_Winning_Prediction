# IPL Match Winning Probability Prediction using ANN

## Open in Google Colab
1. Download `IPL_Match_Winning_Prediction_ANN.ipynb`.
2. Open https://colab.research.google.com/
3. Select **File > Upload notebook** and upload the notebook.
4. Run cells from top to bottom. The notebook downloads the IPL JSON archive from Cricsheet.

## Project target
Predict the probability that the chasing team wins, using second-innings match-state snapshots after each delivery.

## Main tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras.

## Important methodology
- Grouped train/validation/test split by match ID to reduce leakage across snapshots from the same match.
- Numeric standardization and categorical one-hot encoding are fitted only on training data.
- The test set is held out for final evaluation.
- D/L matches are excluded from the baseline due to revised target complexity.

## Data source
Cricsheet: https://cricsheet.org/downloads/

Review Cricsheet's data notes and terms before redistributing data.
