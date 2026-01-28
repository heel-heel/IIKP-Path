# IDKP-Path

## Repository Structure
- `Datasets/`: All datasets. You can obtain the well-preprocessed datasets from [[Baidu Drive]](Link for modify)
    - In each dataset, You can get: 
        - `Data_Quality/`: This records the results of data quality evaluation on this dataset, including: KS Test, KL Divergence, 2-Wasserstein Distance, Sliced Wasserstein Distance, and Mutual Information.
        - `Imputation/`: This records the imputed datasets.
        - `Mechanism/`: This records the results of key factors analysis.
        - `null/`: This records the dataset after missing value injection.
        - `clean.csv`: This is the original clean data.
- `Imputation_Algorithms/`: The imputation algorithms and their running time.
    - `Categorical/`: The imputation algorithms for categorical attributes.
    - `Numerical/`: The imputation algorithms for numerical attributes.
    - `Time_Record/`: The running time results. 
- `Imputation_Analysis/`: Imputation analysis and the results.
    - `Results_numerical/`: The results of imputation analysis.
    - `hdi_analysis_numerical.py`: The imputation analysis for HDI.
    - `knn_analysis_numerical.py`: The imputation analysis for KNN.
    - `mice_analysis_numerical.py`: The imputation analysis for MICE.
    - `si_analysis_numerical.py`: The imputation analysis for SI.
- `Data_Quality/`: Code for data quality assessment, including: KS Test, KL Divergence, 2-Wasserstein Distance, Sliced Wasserstein Distance, and Mutual Information.
- `Mechanism/`: Code for key factors analysis
    - `classification/`: Key factors analysis for classification.
        - `factors_relation_results/`: The relationship of class discriminability & label correctness ratio.
        - `class_discriminability/`: Calculate the class discriminability of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`.
        - `discriminability_precision/`: The KFBF algorithm, get the upper and lower bound.
        - `discriminability_precision_random/`: Get the random generated dataset(random class discriminability of the fixed label correctness ratio).
        - `factors_relation/`: Calculate the relationship of class discriminability & label correctness ratio, and the results will be saved to the `Mechanism/ classification/ factors_relation_results`.
        - `label_correctness_ratio/`: Calculate the label correctness ratio of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`.
    - `regression/`: Key factors analysis for regression
        - `factors_relation_results/`: The relationship of feature-target correlation & imputation bias.
        - `factors_relation/`: Calculate the relationship of feature-target correlation & imputation bias, and the results will be saved to the `Mechanism/ regression/ factors_relation_results`.
        - `feature_target_corr/`: Calculate the feature-target correlation of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`.
        - `imputation_deviation/`: Calculate the imputation bias of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`.
    - `timeseries/`: Key factors analysis for time series forecasting.
        - `decompose_basic/`
            - `decompose_basic.py`: Perform basic seasonal decomposition on time series.
        - `decompose_change_bad/`
            - `resid.py`: The ingredient_O1 and ingredient_O2 are processed, ingredient_T is residual. 
            - `seasonal.py`: The ingredient_O1 and ingredient_O2 are processed, ingredient_T is seasonality.
            - `trend.py`: The ingredient_O1 and ingredient_O2 are processed, ingredient_T is trend.
        - `decompose_change_good/`
            - `resid.py`: The ingredient_O1 and ingredient_O2 are unprocessed, ingredient_T is residual.
            - `seasonal.py`: The ingredient_O1 and ingredient_O2 are unprocessed, ingredient_T is seasonality.
            - `trend.py`: The ingredient_O1 and ingredient_O2 are unprocessed, ingredient_T is trend.
        - `decompose_replace/`
            - `resid.py`: Replace the residual.
            - `seasonal.py`: Replace the seasonality.
            - `trend.py`: Replace the trend.
- `Downstream_Tasks`: The code for performance evaluation.
    - `classification/`: The classification Tasks.
        - `test_discriminability_precision.py`: Evaluate the performance of datasets corresponding to the upper and lower bounds of class discriminability.
        - `test_discriminability_precision_random.py`: Evaluate the performance of the randomly generated datasets during the process of calculating the upper and lower bounds of class discriminability.
        - `test_imputation.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets.
        - `test_imputation_forPruningTest.py`: Evaluation the performance of datasets that is for pruning test, including the original dataset, history dataset and test dataset. 
    - `regression/`: The regression tasks.
        - `test_imputation.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets.
        - `test_imputation_forPruningTest`: Evaluation the performance of datasets that is for pruning test, including the original dataset, history dataset and test dataset. 
    - `timeseries/`: The time series forecasting tasks.
        - `layers`: It helps the implementation of TimeMixer. And the code is from [[TSLib]](https://github.com/thuml/Time-Series-Library)
        - `test_decompose_change_bad.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are processed. The model is MLP.
        - `test_decompose_change_bad_LightTS.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are processed. The model is LightTS.
        - `test_decompose_change_bad_TimeMixer.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are processed. The model is TimeMixer.
        - `test_decompose_change_bad_TSMixer.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are processed. The model is TSMixer.
        - `test_decompose_change_good.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are unprocessed. The model is MLP.
        - `test_decompose_change_good_LightTS.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are unprocessed. The model is LightTS.
        - `test_decompose_change_good_TimeMixer.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are unprocessed. The model is TimeMixer.
        - `test_decompose_change_good_TSMixer.py`: Evaluate the performance of datasets whose ingredient_O1 and ingredient_O2 are unprocessed. The model is TSMixer.
        - `test_decompose_replace.py`: Evaluate the performance of datasets whose ingredient has been replaced. The model is MLP.
        - `test_decompose_replace_LightTS.py`: Evaluate the performance of datasets whose ingredient has been replaced. The model is LightTS.
        - `test_decompose_replace_TimeMixer.py`: Evaluate the performance of datasets whose ingredient has been replaced. The model is TimeMixer.
        - `test_decompose_replace_TSMixer.py`: Evaluate the performance of datasets whose ingredient has been replaced. The model is TSMixer.
        - `test_imputation.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets. The model is MLP.
        - `test_imputation_forPruningTest.py`: Evaluation the performance of datasets that is for pruning test, including the original dataset, history dataset and test dataset. The model is MLP.
        - `test_imputation_LightTS.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets. The model is LightTS.
        - `test_imputation_TimeMixer.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets. The model is TimeMixer.
        - `test_imputation_TSMixer.py`: Evaluation the performance of clean datasets, missing datasets and imputed datasets. The model is TSMixer.
- `Downstream_Results/`: The resuls of performance evaluation.
- `RunScipts/`
    - `ForPruningTest/`: The scripts for pruning test.
        - `run_imputation_categorical_forPruningTest.py`: Impute the categorical attributes of datasets that are for pruning test.
        - `run_imputation_numerical_forPruningTest.py`: Impute the numerical attributes of datasets that are for pruning test.
        - `run_insert_null_forPruningTest.py`: Insert missing values into datasets that are for pruning test.
    - `run_data_quality.py`: The scripts for data quality assessment.
    - `run_imputation_categorical.py`: Impute the categorical attributes of datasets.
    - `run_imputaiton_numerical.py`: Impute the numerical attributes of datasets.
    - `run_insert_null.py`: Insert missing values into datasets.
- `util/`: Some figures.


## Parameters
### Time Series Forecasting         
#### MLP
| dataset      | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha  | activation | look_back |    
|--------------|--------|----------|-------------------|--------------------|----------------|--------|------------|-----------|
| M4-Daily     | sgd    | 3000     | 0.1               | (75,)              | True           | 0.01   | tanh       | 6         |
| M4-Weekly    | sgd    | 1000     | 0.1               | (50,20)            | True           | 0.1    | tanh       | 13        |
| M4-Monthly   | sgd    | 1000     | 0.1               | (80, 30)           | True           | 0.01   | tanh       | 13        |
| M4-Quarterly | adam   | 3000     | 0.01              | (50, 20, 10)       | True           | 0.001  | tanh       | 12        |
| M4-Yearly    | sgd    | 3000     | 0.001             | (60, 20)           | True           | 0.01   | relu       | 9         | 
| M3-Yearly    | sgd    | 1000     | 0.1               | (50, 30)           | True           | 0.001  | tanh       | 6         | 
| ETTh1        | sgd    | 1000     | 0.1               | (100, 50, 30)      | True           | 0.001  | relu       | 20        | 
| ETTm1        | sgd    | 1000     | 0.1               | (100, 50)          | True           | 0.001  | relu       | 6         | 
| Illness      | adam   | 1000     | 0.1               | (100,)             | True           | 0.001  | relu       | 10        | 
| Exchange     | adam   | 1000     | 0.1               | (100, 50)          | True           | 0.001  | relu       | 8         | 
| Weather      | adam   | 1000     | 0.1               | (80,)              | True           | 0.001  | relu       | 5         | 
| ETTh2        | sgd    | 1000     | 0.1               | (50,)              | True           | 0.0001 | tanh       | 10        | 
 


#### TSMixer
| dataset      | num_epochs | e_layers | d_model | dropout | early_stopping |    
|--------------|------------|----------|---------|---------|----------------|
| M4-Daily     | 70         | 3        | 15      | 0.15    | False          |
| M4-Weekly    | 30         | 3        | 25      | 0.3     | False          |
| M4-Monthly   | 200        | 3        | 15      | 0.15    | False          |
| M4-Quarterly | 80         | 3        | 20      | 0.15    | False          |
| M4-Yearly    | 100        | 3        | 15      | 0.25    | False          |
| ETTh1        | 80         | 2        | 50      | 0.3     | False          |
| ETTm1        | 30         | 2        | 20      | 0.1     | False          |
| Illness      | 20         | 2        | 15      | 0.15    | False          |
| Exchange     | 1000       | 1        | 15      | 0.1     | False          |
| Weather      | 1000       | 1        | 15      | 0.2     | False          |



#### LightTS
| dataset      | num_epochs | d_model | early_stopping | learning_rate |  
|--------------|------------|---------|----------------|---------------|
| M4-Daily     | 1000       | 68      | False          | 0.0001        |
| M4-Weekly    | 1000       | 64      | False          | 0.0001        |
| M4-Monthly   | 270        | 72      | False          | 0.0001        |
| M4-Quarterly | 300        | 72      | False          | 0.0001        |
| M4-Yearly    | 200        | 64      | False          | 0.0001        |
| ETTh1        | 1000       | 68      | False          | 0.0001        |
| ETTm1        | 1000       | 60      | True           | 0.0001        |
| Illness      | 100        | 128     | True           | 0.01          |
| Exchange     | 100        | 64      | True           | 0.01          |
| Weather      | 500        | 92      | True           | 0.0001        |


#### TimeMixer
| dataset      | num_epochs | e_layer | d_model | dropout | early_stopping |  
|--------------|------------|---------|---------|---------|----------------|
| M4-Daily     | 6          | 4       | 20      | 0.2     | False          |
| M4-Weekly    | 5          | 4       | 32      | 0.1     | False          |
| M4-Monthly   | 5          | 4       | 64      | 0.35    | False          |
| M4-Quarterly | 5          | 4       | 28      | 0.25    | False          |
| M4-Yearly    | 5          | 4       | 24      | 0.25    | False          |
| ETTh1        | 15         | 4       | 32      | 0.1     | False          |
| ETTm1        | 15         | 2       | 36      | 0.1     | False          |
| Illness      | 50         | 2       | 32      | 0.1     | False          |
| Exchange     | 20         | 2       | 32      | 0.15    | False          |
| Weather      | 40         | 3       | 16      | 0.1     | False          |



### Classification   
| dataset         | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha  | activation |    
|-----------------|--------|----------|-------------------|--------------------|----------------|--------|------------|
| Beers           | adam   | 1000     | 0.001             | (50, 20)           | False          | 0.0001 | relu       |          
| Flights         | adam   | 1000     | 0.001             | (100,)             | False          | 0.0001 | relu       |           
| Hospital        | adam   | 1000     | 0.001             | (50,)              | False          | 0.0001 | relu       |
| RedWineQuality  | adam   | 1000     | 0.0001            | (100,)             | False          | 0.0001 | relu       |
| AvocadoRipeness | adam   | 1000     | 0.1               | (50,)              | False          | 0.1    | relu       |
| Glass           | adam   | 1000     | 0.0001            | (50, 20)           | False          | 0.1    | relu       |


### Regression
| dataset          | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha  | activation |    
|------------------|--------|----------|-------------------|--------------------|----------------|--------|------------| 
| concrete         | sgd    | 1000     | 0.001             | (80,40,10)         | True           | 0.1    | relu       |    
| CCPP             | sgd    | 1000     | 0.1               | (90,)              | True           | 0.1    | relu       |    
| AirfoilSelfNoise | adam   | 1000     | 0.001             | (100,50,20)        | True           | 0.001  | relu       |   
| Abalone          | adam   | 1000     | 0.1               | (100,50)           | True           | 0.1    | relu       |   
| ParisHousing     | sgd    | 1000     | 0.1               | (100,)             | True           | 0.1    | relu       |  
| BostonHousePrice | sgd    | 8000     | 0.0001            | (120,)             | False          | 0.0001 | relu       |  


## Environment
```bash
conda env create -f environment.yml
conda activate IIKP_Path
```


## Other
DataClenaing/Downstream_Tasks/timeseries/layers来自论文《Deep Time Series Models:  A Comprehensive Survey and Benchmark》，它的github链接为 https://github.com/thuml/Time-Series-Library        