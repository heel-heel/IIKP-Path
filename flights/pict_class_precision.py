import pandas as pd
import os

# 定义模型名称和缺失比例
model_names = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]

# 初始化结果列表
results = []

# 读取干净数据
clean_df = pd.read_csv('clean.csv')

save_path='./class_quality/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for model_name in model_names:
    for percentage in percentages:
        dirty_file = f'dirty-flight-{percentage}.csv'
        filled_file = f'dirty-flight-{model_name}-{percentage}.csv'
        read_path_null = f'null-flight'
        read_path = f'null-flight-{model_name}'
        dirty_df = pd.read_csv(os.path.join(read_path_null,dirty_file))
        filled_df = pd.read_csv(os.path.join(read_path,filled_file))

        consistent_count = 0
        missing_total = 0

        missing_positions = dirty_df['flight'].isnull()
        for row in range(missing_positions.shape[0]):
            if missing_positions[row]:
                missing_total += 1
                if filled_df.at[row, 'flight'] == clean_df.at[row, 'flight']:
                    consistent_count += 1

        consistent_rate = consistent_count / missing_total if missing_total > 0 else 0

        # 将结果保存到列表中
        results.append({
            'filled_file':filled_file,
            'Consistent Count': consistent_count,
            'Missing Total': missing_total,
            'Consistent Rate': consistent_rate
        })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path,'filling_precision_results.csv'), index=False)
print("结果已保存到filling_precision_results.csv")