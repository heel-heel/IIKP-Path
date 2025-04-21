# DataCleaning

##参数组合
time series forecasting         
  
|dataset       |solver            |max_iter          |leaning_rate_init |hidden_layer_sizes|early_stopping|alpha       |activation |    
|--------------|------------------|------------------|------------------|------------------|--------------|------------|-----------|
|M4-Monthly    |sgd               |1000              |0.1               |(80, 30)          |True          |0.01        |tanh       |
|M4-Quarterly  |adam              |3000              |0.01              |(50, 20, 10)      |True          |0.001       |tanh       |
|M4-Yearly     |sgd               |3000              |0.001             |(60, 20)          |True          |0.01        |relu       | 
