# Binary Classification to Predict Reccurence of Thyroid Cancer

[Adam David](https://www.adamdavid.dev)


---


## Project Overview  
This project aims to predict the recurrence of differentiated thyroid cancer using a feedforward neural network. The model performs binary classification using TensorFlow with the Keras API.

The dataset used is the **Differentiated Thyroid Cancer Recurrence** dataset from the UCI Machine Learning Repository. The model learns to classify patients based on multiple features, including numerical, binary, ordinal, and nominal categorical data.

---

## Dataset  
**Link:** [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)  

### Features  
1. **Numerical:**  
    - Age  
2. **Categorical:**  
    - **Binary:** Gender, Smoking, Hx Smoking, Hx Radiotherapy, Adenopathy, Focality  
    - **Ordinal:** Thyroid Function, Risk, Stage, Response  
    - **Nominal:** Physical Examination, Pathology, T, N, M  

### Target  
The last column of the CSV file is the target variable, indicating whether the cancer has recurred:  
- **0:** No recurrence  
- **1:** Recurrence  

## Project Structure  
This project contains the following key components:  
- **Preprocessing:**  
    - Converts binary features to 0/1 values.  
    - Ordinal encoding for features with inherent order.  
    - Label encoding for nominal features.  
    - Standardization of numerical features.  
- **Neural Network Architecture:**  
    - **Input Layer**: 16 neurons (representing the 16 input features)  
    - **Hidden Layers**:  
        - 64 neurons, ReLU activation  
        - 32 neurons, ReLU activation  
        - 16 neurons, ReLU activation  
        - 4 neurons, ReLU activation  
    - **Output Layer**:  
        - 1 neuron, Sigmoid activation (for binary classification)  
    - **Optimizer**: Stochastic Gradient Descent (SGD)  
    - **Loss Function**: Binary Cross-Entropy  
    - **Metric**: Accuracy
- **Model Training and Evaluation:**  
    - Uses 80-20 train-test split.  
    - Performance metrics include Accuracy and Loss.  

## Installation and Requirements  

```bash
# Clone the repository
git clone https://github.com/adxmd/thyroid-cancer-recurrence.git

# Navigate to the project directory
cd thyroid-cancer-recurrence

# Install the required dependencies
pip install -r requirements.txt
pip install tensorflow scikit-learn pandas matplotlib
```

## How to Run

1. Place the dataset CSV file in your project directory
2. Modify the file path in the script accordingly
3. Run the script using: 


```bash
python3 thyroid_cancer.py
```


## Results

This project includes a function called `TCR_experiments()` that evaluates the models performance over different combinations of learning rates and training epochs. 
- `Learning Rates: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]`
- `Epochs: [5,10,15,20,25,30,35,40]`

Visualizations are generated using `Matplotlib` to compare how the accuracy and loss are affected by the learning rate and how many epochs the model is trained for. 

<!-- ![alt text](/thyroid-cancer-recurrence/results_moreLR.png) -->
![alt text](https://github.com/adxmd/thyroid-cancer-recurrence/blob/main/results_moreLR.png?raw=true)

Based on this visualization we can conclude that for this current neural network architecture, `Learning Rates: 0.01, 0.05, and 0.1` provide the best accuracy and loss values. They can predict recurrence with **~95% accuracy** after training for 40 epochs



