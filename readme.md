# Feed Forward Neural Network (FNN) Implementation

A from-scratch implementation of a feedforward neural network in Python using only NumPy for MNIST digit classification.

## Setup Instructions

### 1. Data Preparation
1. Download the MNIST dataset archive
2. Unzip the `archive_mnist.zip` file in the project directory
3. Ensure the following files are present:
   - `archive_mnist/mnist_train.csv`
   - `archive_mnist/mnist_test.csv`

### 2. Dependencies
```bash
pip install pandas numpy matplotlib
```

## FNN Class Documentation

### Architecture
The `FNN` class implements a fully connected neural network with the following features:
- **Activation Functions**: ReLU for hidden layers, Softmax for output layer
- **Weight Initialization**: He initialization with fan-in scaling
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Regularization**: Gradient clipping

### Class Methods

#### `__init__(self, structure)`
**Purpose**: Initialize the neural network with specified architecture

**Parameters**:
- `structure` (list): Layer sizes, e.g., `[784, 100, 10]`

**Matrix Dimensions**:
- **Weights** `w[i]`: `(structure[i], structure[i+1])`
- **Biases** `b[i]`: `(structure[i+1], 1)`

**Example**:
```python
network = FNN([784, 100, 10])
# Creates:
# w[0]: (784, 100) - Input to hidden layer weights
# b[0]: (100, 1)   - Hidden layer biases
# w[1]: (100, 10)  - Hidden to output layer weights  
# b[1]: (10, 1)    - Output layer biases
```

#### `forward_propagation(self, X)`
**Purpose**: Compute forward pass through the network

**Input Matrix Dimensions**:
- `X`: `(784, 1)` for single sample

**Output**:
- `Z`: List of pre-activation values for each layer
- `A`: List of post-activation values for each layer

**Matrix Flow**:
```
Input X: (784, 1)
├── Layer 1: W₁ᵀ @ X + b₁ → Z₁: (100, 1) → ReLU → A₁: (100, 1)
└── Layer 2: W₂ᵀ @ A₁ + b₂ → Z₂: (10, 1) → Softmax → A₂: (10, 1)
```

#### `backward_propagation(self, Z, A, y, learning_rate)`
**Purpose**: Compute gradients and update weights using backpropagation

**Input Matrix Dimensions**:
- `y`: `(10, 1)` - One-hot encoded target
- `Z`, `A`: From forward propagation

**Gradient Computations**:
- **Output Layer**: `δ₂ = A₂ - y` → `(10, 1)`
- **Hidden Layer**: `δ₁ = (W₂ @ δ₂) ⊙ ReLU'(Z₁)` → `(100, 1)`
- **Weight Gradients**: `∂W₁ = A₀ @ δ₁ᵀ` → `(784, 100)`
- **Bias Gradients**: `∂b₁ = δ₁` → `(100, 1)`

#### `train(self, X, Y, learning_rate=0.1)`
**Purpose**: Train the network on a single sample

**Input Requirements**:
- `X`: `(784, 1)` - Single MNIST image (flattened and normalized)
- `Y`: `(10, 1)` - One-hot encoded label

#### `predict(self, X)`
**Purpose**: Make predictions on input data

**Returns**: Integer class prediction (0-9 for MNIST)

## Usage Examples

### Basic Training
```python
# Load and preprocess data
test = pd.read_csv("./archive_mnist/mnist_test.csv")
train = pd.read_csv("./archive_mnist/mnist_train.csv")

test_x = np.array(test.iloc[:, 1:])/255    # Shape: (10000, 784)
test_y = np.array(pd.get_dummies(test.iloc[:, 0]))  # Shape: (10000, 10)
train_x = np.array(train.iloc[:, 1:])/255  # Shape: (60000, 784) 
train_y = np.array(pd.get_dummies(train.iloc[:, 0])) # Shape: (60000, 10)

# Create network
network = FNN([784, 100, 10])

# Train on single samples
for i in range(1000):
    x_single = train_x[i:i+1].T  # Shape: (784, 1)
    y_single = train_y[i:i+1].T  # Shape: (10, 1)
    network.train(x_single, y_single, learning_rate=0.01)

# Make predictions
test_sample = test_x[0:0+1].T  # Shape: (784, 1)
prediction = network.predict(test_sample)
print(f"Predicted digit: {prediction}")
```

### Architecture Examples

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| `[784, 10]` | Single layer | Simple baseline |
| `[784, 100, 10]` | One hidden layer | Standard MNIST |
| `[784, 128, 64, 10]` | Two hidden layers | Complex patterns |
| `[784, 256, 128, 64, 10]` | Deep network | Advanced learning |

## Data Specifications

### MNIST Dataset Structure
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images  
- **Image dimensions**: 28×28 pixels (flattened to 784 features)
- **Classes**: 10 digits (0-9)
- **Pixel values**: Normalized to [0, 1] range

### Matrix Dimensions Summary

| Component | Single Sample |
|-----------|---------------|
| **Input X** | `(784, 1)` |
| **Target Y** | `(10, 1)` |
| **Hidden Layer** | `(100, 1)` |
| **Output Layer** | `(10, 1)` |
| **Weights W₁** | `(784, 100)` |
| **Weights W₂** | `(100, 10)` |

## Training Monitoring

Use the included `track_accuracy_only()` function to monitor training progress:

```python
steps, accuracies = track_accuracy_only(
    network, train_x, train_y, test_x, test_y,
    num_samples=5000, interval=50, test_size=200
)
```

## Performance Notes

- **Expected Accuracy**: 85-92% on MNIST test set
- **Training Time**: ~5-10 seconds per 1000 samples
- **Memory Usage**: Minimal (network stores only weights and biases)
- **Convergence**: Typically within 2000-5000 training samples

## Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| **Learning Rate** | 0.005 - 0.01 | Lower for stability |
| **Architecture** | `[784, 100, 10]` | Good balance |
| **Training Samples** | 5000+ | For convergence |
| **Gradient Clipping** | [-0.5, 0.5] | Prevents explosion |

## File Structure
```
project/
├── backpropagation.ipynb    # Main implementation
├── archive_mnist/           # Dataset directory
│   ├── mnist_train.csv      # Training data (109 MB)
│   └── mnist_test.csv       # Test data (18 MB)
└── README.md               # This file
```

## Troubleshooting

**Common Issues**:
1. **Shape mismatch**: Ensure input is transposed to column vector `(784, 1)`
2. **Poor convergence**: Try lower learning rate or better initialization
3. **Memory errors**: Reduce number of training samples

**Performance Tips**:
- Start with simple architecture `[784, 10]` to verify implementation
- Use learning rate scheduling for better convergence
- Monitor accuracy curves to detect overfitting or