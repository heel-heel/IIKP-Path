# DataCleaning

## 参数组合
### time series forecasting         
#### mlp     

| dataset      | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha | activation | look_back |    
|--------------|--------|----------|-------------------|--------------------|----------------|-------|------------|-----------|
| M4-Hourly    |        |          |                   |                    |                |       |            |           |
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


### classification   
| dataset         | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha  | activation |    
|-----------------|--------|----------|-------------------|--------------------|----------------|--------|------------|
| Beers           | adam   | 1000     | 0.001             | (50, 20)           | False          | 0.0001 | relu       |          
| Flights         | adam   | 1000     | 0.001             | (100,)             | False          | 0.0001 | relu       |           
| Hospital        | adam   | 1000     | 0.001             | (50,)              | False          | 0.0001 | relu       |
| RedWineQuality  | adam   | 1000     | 0.0001            | (100,)             | False          | 0.0001 | relu       |
| AvocadoRipeness | adam   | 1000     | 0.1               | (50,)              | False          | 0.1    | relu       |
| Glass           | adam   | 1000     | 0.0001            | (50, 20)           | False          | 0.1    | relu       |



### regression
| dataset          | solver | max_iter | leaning_rate_init | hidden_layer_sizes | early_stopping | alpha  | activation |    
|------------------|--------|----------|-------------------|--------------------|----------------|--------|------------|
| M4-Monthly       | sgd    | 8000     | 0.0001            | (100,)             | False          | 0.001  | relu       |  
| M4-Quarterly     | sgd    | 5000     | 0.0001            | (100,)             | True           | 0.001  | relu       |
| M4-Yearly        | sgd    | 5000     | 0.0001            | (50,)              | True           | 0.0001 | relu       |    
| concrete         | sgd    | 1000     | 0.001             | (80,40,10)         | True           | 0.1    | relu       |    
| CCPP             | sgd    | 1000     | 0.1               | (90,)              | True           | 0.1    | relu       |    
| AirfoilSelfNoise | adam   | 1000     | 0.001             | (100,50,20)        | True           | 0.001  | relu       |   
| Abalone          | adam   | 1000     | 0.1               | (100,50)           | True           | 0.1    | relu       |   
| ParisHousing     | sgd    | 1000     | 0.1               | (100,)             | True           | 0.1    | relu       |  
| BostonHousePrice | sgd    | 8000     | 0.0001            | (120,)             | False          | 0.0001 | relu       |  



## 关于Imputation_Strategy_Selection/selection_pruning.py的使用说明
python selection_pruning.py <task_type> <dataset_file>    
如：python selection_pruning.py regression 
要求在所有数据集上都能达到要求

## environment
torch110、myenv
env_timeseries用来处理时间序列预测的MLP变种
使用conda安装（推荐用于科学计算包）
conda install pandas numpy scipy matplotlib scikit-learn tensorflow
使用pip安装其他包
pip install rich tqdm argparse


## 其他
DataClenaing/Downstream_Tasks/timeseries/layers来自论文《Deep Time Series Models:  A Comprehensive Survey and Benchmark》，它的github链接为 https://github.com/thuml/Time-Series-Library        


数据集M3-Yearly、Glass、BostonHousePrice分别用于时序预测、分类、回归

数据集M3-Yearly-history的90%、95%缺失MICE无法收敛，用85%的数据集替代上述2个数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集BostonHousePrice-history的95%缺失MICE无法收敛，用90%的数据集替代上述数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集Glass-history的85%、90%、95%缺失MICE无法收敛，用80%的数据集替代上述3个数据集以保证代码不做过多修改，修改数据不做参考，作为历史数据可以考虑
数据集Glass-test的95%缺失MICE无法收敛，用90%的数据集替代上述数据集以保证代码不做过多修改，修改数据不做参考，作为测试数据不考虑