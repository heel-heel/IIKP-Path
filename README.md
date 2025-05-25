# DataCleaning

## 参数组合
### time series forecasting         
#### mlp     

|dataset       |solver            |max_iter          |leaning_rate_init |hidden_layer_sizes|early_stopping|alpha       |activation |look_back    |    
|--------------|------------------|------------------|------------------|------------------|--------------|------------|-----------|-------------|
|M4-Monthly    |sgd               |1000              |0.1               |(80, 30)          |True          |0.01        |tanh       |13           |
|M4-Quarterly  |adam              |3000              |0.01              |(50, 20, 10)      |True          |0.001       |tanh       |12           |
|M4-Yearly     |sgd               |3000              |0.001             |(60, 20)          |True          |0.01        |relu       |9            | 

#### TSMixer

|dataset       |num_epochs        |e_layers           |d_model           |dropout           |early_stopping|    
|--------------|------------------|------------------|------------------|------------------|---------------|
|M4-Monthly    |200               |3                 |15                |0.15              |False          |
|M4-Quarterly  |80                |3                 |20                |0.15              |False          |
|M4-Yearly     |100               |3                 |15                |0.25              |False          |

#### LightTS

|dataset       |num_epochs        |d_model           |early_stopping    |learning_rate     |  
|--------------|------------------|------------------|------------------|------------------|
|M4-Monthly    |270               |72                |False             |0.0001            |
|M4-Quarterly  |300               |72                |False             |0.0001            |
|M4-Yearly     |200               |64                |False             |0.0001            |

####TimeMixer

|dataset       |num_epochs        |e_layer           |d_model           |dropout           |early_stopping    |  
|--------------|------------------|------------------|------------------|------------------|------------------|
|M4-Monthly    |5                 |4                 |64                |0.35              |False             |
|M4-Quarterly  |5                 |4                 |28                |0.25              |False             |
|M4-Yearly     |5                 |4                 |24                |0.25              |False             |


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

##environment
torch110、myenv
env_timeseries用来处理时间序列预测的MLP变种