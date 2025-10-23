# InterIIT-Sample-PS

## Task 1 - Docker Setup

### Overview
This project demonstrates running a Pathway application using Docker. Since Pathway only works on Linux, Docker allows Windows and Mac users to run Pathway applications seamlessly.


### File Structure
. \
├── pathway_app.py \
├── Dockerfile \
└── requirements.txt


### How to Run

#### Verify Docker Installation
```bash
docker --version
```

#### Build Docker Image
```bash
docker build -t pathway-demo
```

#### Run the Container
```bash
docker run --rm -v ${PWD}:/app -w /app pathwaycom/pathway python sample.py 
```

### Expected Output
```
            | name    | next_age
^YYKZT05... | Alice   | 24
^BDKDDNX... | Bob     | 31
^92RVW3Y... | Charlie | 28
```

## Task 2 - Real time AI-driven stock price prediction

## 📊 Project Overview

This project implements a time series forecasting model that predicts stock prices using historical data. The model uses a combination of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers to capture temporal patterns in stock price movements.

## 🎯 Model Performance

![Model Performance](path/to/your/plot.png)

**Direction Accuracy:** ~XX% (predicted price movement direction)

The model successfully predicts whether the stock price will go up or down compared to the previous day's price.

## 📁 Project Structure

```
.
├── data/
│   ├── btc_historical.csv      # Raw historical stock data
│   └── processed_data.npz      # Preprocessed training/testing data
├── models/
│   ├── best_model.keras        # Trained LSTM-GRU model
│   ├── scaler_X.pkl           # Feature scaler
│   └── scaler_y.pkl           # Target scaler
└── notebooks/
    ├── preprocess.ipynb        # Data preprocessing notebook
    └── train.ipynb            # Model training notebook
```

## 🔧 Features

### Input Features
- **Average Price**: Mean of Open and Close prices
- **Price Change**: Absolute difference between Close and Open
- **Price Change Percentage**: Relative price change

### Model Architecture
- **Bidirectional LSTM Layer** (64 units) - Captures patterns in both directions
- **GRU Layers** (64 and 32 units) - Efficient temporal processing
- **Dropout Layers** - Prevents overfitting
- **Dense Layers** - Final prediction layers

### Technical Specifications
- **Window Size**: 30 time steps
- **Train/Test Split**: 80/20
- **Optimizer**: Adam with learning rate 5e-4
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience of 30 epochs

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib
```

### Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd stock-price-prediction
```

2. Ensure your data file is in the correct location:
   - Place your CSV file in `data/btc_historical.csv`
   - CSV should have columns: `Date`, `Open`, `Close`

### Usage

#### 1. Data Preprocessing

Run the preprocessing notebook:
```bash
jupyter notebook notebooks/preprocess.ipynb
```

This will:
- Load and clean the raw data
- Create windowed sequences
- Generate technical features
- Scale the data
- Save processed data to `data/processed_data.npz`

#### 2. Model Training

Run the training notebook:
```bash
jupyter notebook notebooks/train.ipynb
```

This will:
- Load preprocessed data
- Build the LSTM-GRU hybrid model
- Train with early stopping and learning rate reduction
- Save the best model to `models/best_model.keras`
- Generate performance plots

## 📈 Model Training Details

### Callbacks Used
- **EarlyStopping**: Stops training if validation loss doesn't improve for 30 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus
- **ModelCheckpoint**: Saves the best model based on validation loss

### Hyperparameters
```python
EPOCHS = 200
BATCH_SIZE = 32
WINDOW_SIZE = 30
SPLIT_RATIO = 0.8
VALIDATION_SPLIT = 0.15
```

## 📊 Evaluation Metrics

- **Direction Accuracy**: Percentage of correct predictions for price movement direction (up/down)
- **MSE Loss**: Mean squared error on validation set
- **Visual Comparison**: Actual vs Predicted price plots

## 🔍 Results Interpretation

The model outputs:
1. **Predicted Prices**: Continuous values for future stock prices
2. **Direction Prediction**: Binary classification (price up or down)
3. **Performance Plots**: Visual comparison of predictions vs actual prices

## 🛠️ Customization

### Changing the Window Size
Edit in preprocessing:
```python
WINDOW_SIZE = 60  # Increase for longer historical context
```

### Modifying Model Architecture
Edit in training notebook:
```python
# Add more layers or change units
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(GRU(64, return_sequences=True))
```

### Adding More Features
Edit in preprocessing:
```python
df["moving_avg_7"] = df["avg_price"].rolling(window=7).mean()
df["volatility"] = df["Close"].rolling(window=7).std()
```

## 📝 Notes

- The model is trained on normalized data (MinMaxScaler with range [0,1])
- Separate scalers are used for features (X) and target (y)
- The model predicts the next time step's average price
- Direction accuracy is often more reliable than absolute price prediction

## ⚠️ Disclaimer

This model is for educational purposes only. Stock price prediction is extremely challenging and past performance does not guarantee future results. Do not use this model for actual trading decisions without thorough validation and risk assessment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: October 2025
