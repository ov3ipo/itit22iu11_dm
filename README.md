# Weka Machine Learning Framework Implementation Report

## 1. Data Loading and Preprocessing

The project leverages Weka's data loading and preprocessing capabilities through several key components:

### Data Loading Strategy
- **CSVLoader**: Used in `DataLoader` class to convert CSV files to Weka's native ARFF format
- **ArffLoader**: Used in `ClassificationFramework` to load processed ARFF files
- **Conversion Process**:
    - CSV files are first loaded using `CSVLoader`
    - Data is then preprocessed (removing unnecessary attributes, setting class index)
    - Data can be saved as ARFF using `ArffSaver`

### Data Preprocessing Techniques
- **Standardization**: Uses Weka's `Standardize` filter to normalize numeric attributes
  ```java
  Standardize standardize = new Standardize();
  standardize.setInputFormat(fullData);
  fullData = Filter.useFilter(fullData, standardize);
  ```
- Automatic attribute removal
- Class index setting
- Optional data sampling for large datasets

## 2. Prediction Framework

### Classifier Abstraction
- **Interfaces and Abstract Classes**:
    - `IModel` interface defines core method signatures
    - `BaseClassifier` provides common implementation for classifier methods
    - Specific classifier classes (LinearRegression, SVMRegression, etc.) extend `BaseClassifier`

### Model Training and Evaluation
- Supports multiple regression models:
    - Linear Regression
    - SVM Regression
    - M5P Decision Tree
    - Random Forest

### Evaluation Methods
1. **Train-Test Split**:
    - 80% training, 20% testing data
    - Manual prediction and error calculation

2. **Cross-Validation**:
    - 10-fold cross-validation using Weka's `Evaluation` class
    - Calculates metrics like:
        * Correlation coefficient
        * Mean absolute error
        * Root mean squared error
        * Relative error percentages

## 3. Model Configuration

Each classifier is configured with specific Weka API parameters:

### Example: SVM Regression Configuration
```java
SMOreg svm = new SMOreg();
svm.setC(1.0);  // Complexity parameter
svm.setFilterType(new SelectedTag(SMOreg.FILTER_NORMALIZE, SMOreg.TAGS_FILTER));

PolyKernel polyKernel = new PolyKernel();
polyKernel.setExponent(1.0);
svm.setKernel(polyKernel);
```

### Example: Linear Regression Configuration
```java
lr.setAttributeSelectionMethod(new SelectedTag(1, LinearRegression.TAGS_SELECTION));
lr.setRidge(1.0E-8);
lr.setEliminateColinearAttributes(true);
```

## 4. User Interface Integration

- **GUI** allows interactive:
    - Dataset loading
    - Classifier selection
    - Model training and evaluation
- Background threading for model training
- Real-time logging of training process

## Key Weka API Highlights
- `Instances`: Core data structure
- `Filter`: Data preprocessing
- `Classifier`: Model training
- `Evaluation`: Performance metrics
- Kernel and optimization configurations

## Conclusion
This implementation demonstrates a comprehensive machine learning workflow using Weka's powerful APIs, providing flexibility in data preprocessing, model training, and evaluation.
