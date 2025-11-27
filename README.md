# IDKP-Path

## Repository Structure
- `Datasets/`: All datasets. You can obtain the well-preprocessed datasets from [[Baidu Drive]](Link for modify)
    - In each dataset, You can get: 
        - `Data_Quality`: This records the results of data quality evaluation on this dataset, including: KS Test, KL Divergence, 2-Wasserstein Distance, Sliced Wasserstein Distance, and Mutual Information.
        - `Imputation`: This records the imputed dataset.
        - `Mechanism`: This records the results of key factors analysis.
        - `null`: This records the dataset after missing value injection.
        - `clean.csv`: This is the original clean data.
- `Data_Quality`: Code for data quality evaluation, including: KS Test, KL Divergence, 2-Wasserstein Distance, Sliced Wasserstein Distance, and Mutual Information.
- `Mechanism`: Code for key factors analysis
    - `classification`: key factors analysis for classification
        - `factors_relation_results`: the relationship of class discriminability & label correctness ratio
        - `class_discriminability`: calculate the class discriminability of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`
        - `discriminability_precision`: the KFBF algorithm, get the upper and lower bound
        - `discriminability_precision_random`: get the random generated dataset(random class discriminability of the fixed label correctness ratio)
        - `factors_relation`: calculate the relationship of class discriminability & label correctness ratio, and the results will be saved to the `Mechanism/ classification/ factors_relation_results`
        - `label_correctness_ratio`: calculate the label correctness ratio of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`
    - `regression`: key factors analysis for regression
        - `factors_relation_results`: the relationship of feature-target_correlation & imputation bias
        - `factors_relation`: calculate the relationship of feature-target_correlation & imputation bias, and the results will be saved to the `Mechanism/ regression/ factors_relation_results`
        - `feature_target_corr`: calculate the feature-target_correlation of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`
        - `imputation_deviation`: calculate the imputation bias of each dataset, and the results will be saved to the `Mechanism` of each dataset in `Datasets`
    - `timeseries`: key factors analysis for time series forecasting
        - `decompose_basic`
            - `decompose_basic.py`: Perform basic seasonal decomposition on time series 
        - `decompose_change_bad`: 
        - `decompose_change_good`:
        - `decompose_replace`:


## Parameters
### Time Series Forecasting         
#### MLP
| dataset      | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha | activation | look_back |    
|--------------|--------|----------|-------------------|--------------------|----------------|-------|------------|-----------|
| M4-Daily     | sgd    | 3000     | 0.1               | (75,)              | True           | 0.01  | tanh       | 6         |
| M4-Weekly    | sgd    | 1000     | 0.1               | (50,20)            | True           | 0.1   | tanh       | 13        |
| M4-Monthly   | sgd    | 1000     | 0.1               | (80, 30)           | True           | 0.01  | tanh       | 13        |
| M4-Quarterly | adam   | 3000     | 0.01              | (50, 20, 10)       | True           | 0.001 | tanh       | 12        |
| M4-Yearly    | sgd    | 3000     | 0.001             | (60, 20)           | True           | 0.01  | relu       | 9         | 
| M3-Yearly    | sgd    | 1000     | 0.1               | (50, 30)           | True           | 0.001 | tanh       | 6         | 

#### TSMixer
| dataset      | num_epochs | e_layers | d_model | dropout | early_stopping |    
|--------------|------------|----------|---------|---------|----------------|
| M4-Daily     | 70         | 3        | 15      | 0.15    | False          |
| M4-Weekly    | 30         | 3        | 25      | 0.3     | False          |
| M4-Monthly   | 200        | 3        | 15      | 0.15    | False          |
| M4-Quarterly | 80         | 3        | 20      | 0.15    | False          |
| M4-Yearly    | 100        | 3        | 15      | 0.25    | False          |

#### LightTS
| dataset      | num_epochs | d_model | early_stopping | learning_rate |  
|--------------|------------|---------|----------------|---------------|
| M4-Daily     | 1000       | 68      | False          | 0.0001        |
| M4-Weekly    | 1000       | 64      | False          | 0.0001        |
| M4-Monthly   | 270        | 72      | False          | 0.0001        |
| M4-Quarterly | 300        | 72      | False          | 0.0001        |
| M4-Yearly    | 200        | 64      | False          | 0.0001        |

#### TimeMixer
| dataset      | num_epochs | e_layer | d_model | dropout | early_stopping |  
|--------------|------------|---------|---------|---------|----------------|
| M4-Daily     | 6          | 4       | 20      | 0.2     | False          |
| M4-Weekly    | 5          | 4       | 32      | 0.1     | False          |
| M4-Monthly   | 5          | 4       | 64      | 0.35    | False          |
| M4-Quarterly | 5          | 4       | 28      | 0.25    | False          |
| M4-Yearly    | 5          | 4       | 24      | 0.25    | False          |


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
conda activate IDKP_Path
```


## Other
DataClenaing/Downstream_Tasks/timeseries/layers来自论文《Deep Time Series Models:  A Comprehensive Survey and Benchmark》，它的github链接为 https://github.com/thuml/Time-Series-Library        
对M4、M4数据集进行处理，使之以列为序列进行处理

数据集M3-Yearly、Glass、BostonHousePrice分别用于时序预测、分类、回归

数据集M3-Yearly-history的90%、95%缺失MICE无法收敛，用85%的数据集替代上述2个数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集BostonHousePrice-history的95%缺失MICE无法收敛，用90%的数据集替代上述数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集Glass-history的85%、90%、95%缺失MICE无法收敛，用80%的数据集替代上述3个数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集Glass-test的95%缺失MICE无法收敛，用90%的数据集替代上述数据集以保证代码不做过多修改，修改数据不做参考，作为测试数据不考虑