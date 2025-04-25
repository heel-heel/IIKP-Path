# DataCleaning

## 参数组合
### time series forecasting         
  
|dataset       |solver            |max_iter          |leaning_rate_init |hidden_layer_sizes|early_stopping|alpha       |activation |    
|--------------|------------------|------------------|------------------|------------------|--------------|------------|-----------|
|M4-Monthly    |sgd               |1000              |0.1               |(80, 30)          |True          |0.01        |tanh       |
|M4-Quarterly  |adam              |3000              |0.01              |(50, 20, 10)      |True          |0.001       |tanh       |
|M4-Yearly     |sgd               |3000              |0.001             |(60, 20)          |True          |0.01        |relu       | 


### classification   
|dataset       |solver            |max_iter          |leaning_rate_init |hidden_layer_sizes|early_stopping|alpha       |activation |    
|--------------|------------------|------------------|------------------|------------------|--------------|------------|-----------|
|Beers         |adam              |1000              |0.001             |(50, 20)          |False         |0.0001      |relu       |          
|Flights       |adam              |1000              |0.001             |(100,)            |False         |0.0001      |relu       |           
|Hospital      |adam              |1000              |0.001             |(50,)             |False         |0.0001      |relu       |


### regression
|dataset       |solver            |max_iter          |leaning_rate_init |hidden_layer_sizes|early_stopping|alpha       |activation |    
|--------------|------------------|------------------|------------------|------------------|--------------|------------|-----------|
|M4-Monthly    |sgd               |8000              |0.0001            |(100,)            |False         |0.001       |relu       |  
|M4-Quarterly  |sgd               |5000              |0.0001            |(100,)            |True          |0.001       |relu       |
|M4-Yearly     |sgd               |5000              |0.0001            |(50,)             |True          |0.0001      |relu       |     

##关于Imputation_Strategy_Selection/selection_pruning.py的使用说明
python selection_pruning.py <task_type> <dataset_file>    
如：python selection_pruning.py regression 
要求在所有数据集上都能达到要求