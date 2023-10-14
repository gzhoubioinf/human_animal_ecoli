
# human animal ecoli



This project is a demonstration of utilizing XGBoost for classification tasks on kmer data. The process involves data pre-processing to transform kmer data into a manageable format, training an XGBoost model, and evaluating its performance.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset Processing](#dataset-processing)
4. [XGBoost Model Training](#xgboost-model-training)
5. [Performance Evaluation](#performance-evaluation)
6. [Contact](#contact)

---

## Installation

Clone this repository to your local machine and navigate to the project directory. Execute the following command to install necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage
### train data and get shap value
Change parameters in input.json file
### Run tools

There are two ways to run the tools

- The package may be downloaded and run by the command
```bash
python main.py input.json
```
- The tool is available on DockerHub and may be fetched and run using the following commands:
```bash
docker pull gzhoubioinf kmer ml:version109
docker run -v $PWD:/data --rm -it kmer ml:version109 ./app.py -i input fasta file -0 /data/output report
```

### data plot 
import kmer_ml_pacakge.visualization to plot data

---

## Dataset Processing

### K-mer Dataset Processing

- **Reading Chunk Files**: There are 2000 chunk files, each containing partial data to be processed. 
- **Generating a Sparse Matrix**: A sparse matrix is generated from the processed chunk files, facilitating efficient storage and computational operations.
- **Filtering Data**: Data filtering is performed based on specified labels to obtain the relevant dataset for model training. The dataset is filtered to remove two designated data types, and the resulting data is categorized into Human & Animal (HA), Human & Human (HH), and Animal & Animal (AA) based on predefined conditions.

---

## XGBoost Model Training

### Grid Search and K-Fold Cross Validation

- **Hyperparameter Tuning**: A grid search is conducted for hyperparameter tuning to find the optimal set of parameters for the XGBoost model.
- **K-Fold Cross Validation**: K-Fold Cross Validation is performed to assess the model's performance. This approach helps to ensure that the model's performance is consistent across different subsets of the data.

---

## Performance Evaluation

### Confidence Intervals for Classification Metrics

- **Calculating Confidence Intervals**: Confidence intervals for classification metrics such as precision, recall, and F1-score are calculated and stored. This provides a range within which the true value of the metric is likely to fall, providing an indication of the model's performance stability.

---

## Contact

- Author: Ge Zhou
- Email: ge.zhou@kaust.edu.sa

---

This documentation provides a high-level overview of the code structure and the project's primary functionalities. For more detailed information, please refer to the inline comments within the script file.

