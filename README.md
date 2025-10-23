# InterIIT-Sample-PS

## Task 1 - Docker Setup

### Overview
This project demonstrates running a Pathway application using Docker. Since Pathway only works on Linux, Docker allows Windows and Mac users to run Pathway applications seamlessly.


### File Structure
```
.
├── pathway_app.py
├── Dockerfile
└── requirements.txt
```

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
docker run --rm -v ${PWD}:/app -w /app pathwaycom/pathway python pathway_app.py 
```

### Expected Output
```
            | name    | next_age
^YYKZT05... | Alice   | 24
^BDKDDNX... | Bob     | 31
^92RVW3Y... | Charlie | 28
```

## Task 2 - Real time AI-driven stock price prediction

### Project Overview

This project implements a time series forecasting model that predicts stock prices using historical data. The model uses a combination of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers to capture temporal patterns in stock price movements.

### Model Performance

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/38fb8e1e-811d-41dd-8023-1525855554d3" />

**Direction Accuracy:** 40.00% 
The accuracy score is defined as the percentage of times the model predicted the direction of the stock (Up or down) correctly.

### Project Structure

```
.
├── data/
│   ├── btc_historical.csv      # Raw historical stock data
│   └── processed_data.npz      # Preprocessed training/testing data
├── models/
│   ├── best_model.keras        # Trained LSTM-GRU model
│   ├── scaler_X.pkl            # Feature scaler
│   └── scaler_y.pkl            # Target scaler
└── notebooks/
    ├── preprocess.ipynb        # Data preprocessing notebook
    └── train.ipynb             # Model training notebook
```

###  Features

#### Input Features
- **Average Price**: Mean of Open and Close prices
- **Price Change**: Absolute difference between Close and Open
- **Price Change Percentage**: Relative price change

#### Model Architecture
- **Bidirectional LSTM Layer** (64 units) - Captures patterns in both directions
- **GRU Layers** (64 and 32 units) - Efficient temporal processing
- **Dropout Layers** - Prevents overfitting
- **Dense Layers** - Final prediction layers

#### Technical Specifications
- **Window Size**: 30 time steps
- **Train/Test Split**: 80/20
- **Optimizer**: Adam with learning rate 5e-4
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience of 30 epochs

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
- Save the scalers to `models/scaler_X.pkl` and `models/scaler_Y.pkl`
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

### Model Training Details

#### Callbacks Used
- **EarlyStopping**: Stops training if validation loss doesn't improve for 30 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus
- **ModelCheckpoint**: Saves the best model based on validation loss

#### Hyperparameters
```python
EPOCHS = 200
BATCH_SIZE = 32
WINDOW_SIZE = 30
SPLIT_RATIO = 0.8
VALIDATION_SPLIT = 0.15
```

### Evaluation Metrics

- **Direction Accuracy**: Percentage of correct predictions for price movement direction (up/down)
- **MSE Loss**: Mean squared error on validation set
- **Visual Comparison**: Actual vs Predicted price plots

### Results Interpretation

The model outputs:
1. **Predicted Prices**: Continuous values for future stock prices
2. **Direction Prediction**: Binary classification (price up or down)
3. **Performance Plots**: Visual comparison of predictions vs actual prices

#### Reason for Low Accuracy
- Limited dataset size and time range  
- High volatility and noise in stock prices  
- Model unable to capture long-term temporal dependencies  
- Lack of external market indicators (news, sentiment, macro data)  
- Overfitting on training data due to small validation set  

### Notes

- The model is trained on normalized data (MinMaxScaler with range [0,1])
- Separate scalers are used for features (X) and target (y)
- The model predicts the next time step's average price
- Direction accuracy is often more reliable than absolute price prediction
