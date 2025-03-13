import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 定义保存路径
save_path = './class_quality/precision_distance'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建日志文件
log_file = os.path.join(save_path, 'log.txt')
results_file = os.path.join(save_path, 'results.txt')

# 清空或创建日志文件和结果文件
with open(log_file, 'w') as f:
    f.write("")

with open(results_file, 'w') as f:
    f.write("CR_Target,J_Target,CR_Actual,J_Actual\n")

# 定义日志记录函数
def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 定义结果记录函数
def save_results(cr_target, j_target, cr_actual, j_actual):
    with open(results_file, 'a') as f:
        f.write(f"{cr_target},{j_target},{cr_actual},{j_actual}\n")

# 读取数据并标准化列名
clean_df = pd.read_csv('clean.csv').rename(columns=str.lower)
dirty_df = pd.read_csv('./null-city/dirty-city-50.csv').rename(columns=str.lower)

# 填补数值型列的缺失值
clean_df['ibu'] = clean_df['ibu'].fillna(clean_df['ibu'].mean())
clean_df['ounces'] = clean_df['ounces'].fillna(clean_df['ounces'].mean())
clean_df['abv'] = clean_df['abv'].fillna(clean_df['abv'].mean())

dirty_df['ibu'] = dirty_df['ibu'].fillna(clean_df['ibu'].mean())
dirty_df['ounces'] = dirty_df['ounces'].fillna(clean_df['ounces'].mean())
dirty_df['abv'] = dirty_df['abv'].fillna(clean_df['abv'].mean())

# 定义目标参数
target_consistent_rates = np.arange(0.1, 1.0, 0.1).tolist()
#j_value_targets = np.arange(0, 5.1, 0.1).tolist()
j_value_targets = [0]
max_iterations = 500
max_cr_error = 0.005
max_J_error = 0.05
consecutive_count = 5  # 连续次数
#change_threshold = 1e-8  # 变化阈值

def find_intermediate_class(class_means, class_A, class_B):
    """找到与两个类平均位置最近的类"""
    mean_AB = (class_means.loc[class_A] + class_means.loc[class_B]) / 2
    distances = np.linalg.norm(class_means - mean_AB, axis=1)
    return class_means.index[np.argmin(distances)]

def get_boundary_samples(class_label, opposite_class, n, features_encoded, target):
    """获取指定类中距离对立类最近的边界样本"""
    mask = (target == class_label)
    if sum(mask) == 0:
        return []
    distances = np.linalg.norm(features_encoded[mask] - class_means.loc[opposite_class], axis=1)
    return features_encoded[mask].index[np.argsort(distances)[:n]]

def calculate_consistent_rate_and_j_value(clean_df, filled_df):
    # 计算填补准确率
    missing_positions = dirty_df['city'].isnull()
    consistent_count = 0
    missing_total = missing_positions.sum()
    for row in range(missing_positions.shape[0]):
        if missing_positions[row]:
            if filled_df.at[row, 'city'] == clean_df.at[row, 'city']:
                consistent_count += 1
    consistent_rate = consistent_count / missing_total if missing_total > 0 else 0

    # 计算 J 值
    features = filled_df.drop(columns=['city'])
    target = filled_df['city']
    features_encoded = pd.get_dummies(features)

    scaler = StandardScaler()
    numerical_features = features.select_dtypes(include=[np.number]).columns
    features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

    unique_cities = target.unique()
    within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
    between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))

    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
        within_class_scatter += class_scatter

    overall_mean = np.mean(features_encoded, axis=0)

    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_size = len(class_data)
        between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

    within_class_distance = np.trace(within_class_scatter)
    between_class_distance = np.trace(between_class_scatter)
    J = between_class_distance / within_class_distance if within_class_distance > 0 else 0

    return consistent_rate, J, features_encoded, target

for cr_target in target_consistent_rates:
    log_message("-"*70)
    log_message("寻找上界......begin......")
    filled_df = dirty_df.copy()
    missing_indices = filled_df[filled_df['city'].isnull()].index
    # 初始填补：按目标CR生成正确/错误值
    correct_size = int(len(missing_indices) * cr_target)
    correct_indices = np.random.choice(missing_indices, size=correct_size, replace=False)
    filled_df.loc[correct_indices, 'city'] = clean_df.loc[correct_indices, 'city']
    error_indices = list(set(missing_indices) - set(correct_indices))
    filled_df.loc[error_indices, 'city'] = np.random.choice(clean_df['city'].unique(), size=len(error_indices))
    max_recent_J_values = 0
    max_recent_J_counter = 0
    max_J_value = 0

    #寻找给定填补正确率的上界
    while True:
        cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
        log_message(f"Now: J={J}, CR={cr}")
        cr_error = cr - cr_target
        #max_J_cr = cr
        max_J_filled_df = filled_df.copy()

        #更新当前上界，并记录上界出现的次数
        if abs(cr_error) <= max_cr_error:
            if J > max_recent_J_values:
                max_recent_J_values = J
                max_J_filled_df = filled_df.copy()
                max_J_cr = cr
                max_recent_J_counter = 0
            else:
                max_recent_J_counter = max_recent_J_counter + 1
                if max_recent_J_counter >= consecutive_count:
                    max_J_value = max_recent_J_values
                    filled_file = os.path.join(save_path, f'filled-max_j-{max_J_value:.3f}-cr-{max_J_cr:3f}.csv')
                    max_J_filled_df.to_csv(filled_file, index=False)
                    log_message("找到上界！")
                    log_message(f"{filled_df}已保存到{save_path}")
                    save_results(cr_target, max_J_value, max_J_cr, max_J_value)
                    break


        # 正确率偏低，需要提高填补正确率
        if cr_error < -max_cr_error:
            log_message("\n微调CR......")
            class_means = features_encoded.groupby(target).mean()
            # 找到所有错误填补的数据点
            error_positions = [
                idx for idx in missing_indices
                if filled_df.at[idx, 'city'] != clean_df.at[idx, 'city']
            ]
            if len(error_positions) == 0:
                log_message("当前没有填补错误的单元格！")
                break
            else:
                # 计算每个错误数据点到其正确类别的距离
                distances_to_correct = []
                for idx in error_positions:
                    correct_class = clean_df.at[idx, 'city']
                    current_features = features_encoded.loc[idx]

                    # 检查正确类别是否在 filled_df 中出现过
                    if correct_class not in filled_df['city'].unique():
                        # 如果未出现过，直接将距离设置为 0
                        distance = 0
                        log_message(f"Class '{correct_class}' not found in filled_df. Setting distance to 0.")
                    else:
                        # 如果出现过，计算实际距离
                        correct_class_mean = class_means.loc[correct_class]
                        distance = np.linalg.norm(current_features - correct_class_mean)

                    distances_to_correct.append((idx, distance))

                n_corrections = min(len(error_positions), int(abs(cr_error) * len(missing_indices)))
                log_message(f"Number of corrections to make: {n_corrections}")

                distances_to_correct.sort(key=lambda x: x[1])  # 按距离从小到大排序

                # 修改选定的数据点为正确类别
                for idx, _ in distances_to_correct[:n_corrections]:
                    filled_df.at[idx, 'city'] = clean_df.at[idx, 'city']
                    log_message(f"Corrected index {idx} to the correct class {clean_df.at[idx, 'city']}")

        # 正确率偏高，需要降低填补正确率
        elif cr_error > max_cr_error:
            log_message("\n微调CR......")
            class_means = features_encoded.groupby(target).mean()
            # 找到所有正确填补的数据点
            correct_positions = [
                idx for idx in missing_indices
                if filled_df.at[idx, 'city'] == clean_df.at[idx, 'city']
            ]
            if len(correct_positions) == 0:
                log_message("No correct positions to corrupt. ERROR!")
            else:
                # 计算每个正确数据点到其正确类别中心的距离
                distances_to_corrupt = []
                for idx in correct_positions:
                    correct_class = clean_df.at[idx, 'city']
                    current_features = features_encoded.loc[idx]
                    correct_class_mean = class_means.loc[correct_class]
                    distance = np.linalg.norm(current_features - correct_class_mean)
                    distances_to_corrupt.append((idx, distance))
                # 计算需要调整的样本数量
                n_corruptions = min(len(correct_positions), int(abs(cr_error) * len(missing_indices)))
                log_message(f"Number of corruptions to make: {n_corruptions}")

                distances_to_corrupt.sort(key=lambda x: x[1], reverse=True)  # 按距离从大到小排序

                # 修改选定的数据点为错误类别
                for idx, _ in distances_to_corrupt[:n_corruptions]:
                    correct_class = clean_df.at[idx, 'city']
                    current_features = features_encoded.loc[idx]
                    # 找到距离最近/最远的错误类别
                    distances_to_other_classes = []
                    for class_name, class_mean in class_means.iterrows():
                        if class_name != correct_class:
                            distance = np.linalg.norm(current_features - class_mean)
                            distances_to_other_classes.append((class_name, distance))
                    distances_to_other_classes.sort(key=lambda x: x[1])

                    # 选择目标类别
                    target_class = distances_to_other_classes[0][0]
                    filled_df.at[idx, 'city'] = target_class
                    log_message(f"Corrupted index {idx} from '{correct_class}' to '{target_class}'")

        #如果cr经过调整，则需重新计算当前的cr和J值
        if abs(cr_error) > max_cr_error:
            log_message("\n微调CR之后......")
            cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
            log_message(f"Now: J={J}, CR={cr}")
            cr_error = cr - cr_target
            if abs(cr_error) <= max_cr_error:
                if J > max_recent_J_values:
                    max_recent_J_values = J
                    max_J_filled_df = filled_df.copy()
                    max_J_cr = cr
                    max_recent_J_counter = 0
                else:
                    max_recent_J_counter = max_recent_J_counter + 1
                    if max_recent_J_counter >= consecutive_count:
                        max_J_value = max_recent_J_values
                        filled_file = os.path.join(save_path, f'filled-max_j-{max_J_value:.3f}-cr-{max_J_cr:.3f}.csv')
                        max_J_filled_df.to_csv(filled_file, index=False)
                        log_message("找到上界！")
                        log_message(f"{filled_df}已保存到{save_path}")
                        save_results(cr_target,max_J_value,max_J_cr,max_J_value)
                        break

        #增大J值
        log_message("\n微调J Value......")
        class_means = features_encoded.groupby(target).mean()
        pairwise_dist = pairwise_distances(class_means)
        np.fill_diagonal(pairwise_dist, np.inf)  # 屏蔽对角线

        # 动态计算调整幅度
        # adjustment_rate = min(0.8, 0.2 * (1 + abs(J_error) * 5))
        adjustment_rate = 0.5

        # 计算每个数据点到所有类中心的距离
        distances_to_means = pairwise_distances(features_encoded, class_means)
        # 找到每个数据点的最临近类别及其距离
        nearest_classes = np.argmin(distances_to_means, axis=1)  # 最临近类别的索引
        nearest_distances = np.min(distances_to_means, axis=1)  # 到最临近类别的距离
        # 获取当前类别标签
        current_classes = target.values
        # 找到当前类别与最临近类别不同的点，并记录其索引和距离
        reclassify_candidates = []
        for idx in range(len(features_encoded)):
            nearest_class_name = class_means.index[nearest_classes[idx]]  # 最临近类别的名称
            current_class_name = current_classes[idx]
            # 确保类别名称的比较是大小写一致的
            if current_class_name.lower() != nearest_class_name.lower():
                reclassify_candidates.append((idx, nearest_distances[idx]))

        # 如果没有符合条件的点，直接跳过
        if len(reclassify_candidates) == 0:
            log_message("No points need reclassification. All points are already in their nearest classes.")
            break
        else:
            # 按距离从小到大排序，并选择距离最小的n个点
            reclassify_candidates.sort(key=lambda x: x[1])  # 按距离排序
            # 根据 reclassify_candidates 的长度动态调整随机数的范围
            if len(reclassify_candidates) >= 10:
                num_to_reclassify = random.randint(2, 10)
            else:
                num_to_reclassify = random.randint(2, len(reclassify_candidates))
            # 分离正确分类和错误分类的点
            correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                        filled_df.at[idx, 'city'] == clean_df.at[idx, 'city']]
            incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                          filled_df.at[idx, 'city'] != clean_df.at[idx, 'city']]

            # 确保至少选择一个正确分类的点和一个错误分类的点
            if correct_class_candidates:
                correct_idx, _ = correct_class_candidates[0]  # 距离最小的正确分类点
            else:
                correct_idx = None

            if incorrect_class_candidates:
                incorrect_idx, _ = incorrect_class_candidates[0]  # 距离最小的错误分类点
            else:
                incorrect_idx = None

            # 如果没有正确分类或错误分类的点，直接选择前 num_to_reclassify 个点
            if correct_idx is None or incorrect_idx is None:
                reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
            else:
                # 合并这两个点
                selected_indices = [correct_idx, incorrect_idx]
                # 确保选择的点数不超过 num_to_reclassify
                remaining_candidates = [idx for idx, _ in reclassify_candidates if idx not in selected_indices]
                additional_indices = remaining_candidates[:num_to_reclassify-2]
                selected_indices.extend(additional_indices)
                reclassify_indices = selected_indices[:num_to_reclassify]

            log_message(f"Reclassifying {len(reclassify_indices)} points to their nearest classes")
            for idx in reclassify_indices:
                current_class_name = current_classes[idx]
                nearest_class_name = class_means.index[nearest_classes[idx]]  # 最临近类别的名称

                # 更新类别
                filled_df.at[idx, 'city'] = nearest_class_name
                log_message(f"Reclassified index {idx} from '{current_class_name}' to '{nearest_class_name}'")

    #根据给定目标值，生成一系列文件
    for j_target in j_value_targets:
        log_message("-" * 70)
        log_message("寻求目标J值......Begin......")
        log_message(f"Target: J={j_target}, CR={cr_target}")
        if (j_target > max_J_value):
            log_message("j_target > max_J_value，跳过")
            break
        filled_df = dirty_df.copy()
        missing_indices = filled_df[filled_df['city'].isnull()].index

        # 初始填补：按目标CR生成正确/错误值
        correct_size = int(len(missing_indices) * cr_target)
        correct_indices = np.random.choice(missing_indices, size=correct_size, replace=False)
        filled_df.loc[correct_indices, 'city'] = clean_df.loc[correct_indices, 'city']
        error_indices = list(set(missing_indices) - set(correct_indices))
        filled_df.loc[error_indices, 'city'] = np.random.choice(clean_df['city'].unique(), size=len(error_indices))

        # 迭代调整
        current_iter = 0
        min_recent_J_bias = 1
        min_J_bias_counter = 0
        min_J_bias_value = 0
        min_J_bias_cr = 0
        min_J_bias_filled_df = filled_df.copy()
        while current_iter < max_iterations:
            log_message("-"*70)
            cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
            log_message(f"Now: J={J}, CR={cr}")
            cr_error = cr - cr_target
            J_error = J - j_target

            #当前正确率在可接受范围内
            if abs(cr_error) <= max_cr_error:
                if abs(J_error) < max_J_error:
                    min_J_bias_value = J
                    min_J_bias_cr = cr
                    min_J_bias_filled_df = filled_df.copy()
                    log_message("当前J值和CR都在可接受范围内！")
                    break
                elif abs(J_error) < min_recent_J_bias:
                    min_recent_J_bias = abs(J_error)
                    min_J_bias_value = J
                    min_J_bias_cr = cr
                    min_J_bias_filled_df = filled_df.copy()
                    min_J_bias_counter = 0
                else:
                    min_J_bias_counter = min_J_bias_counter + 1
                    if (min_J_bias_counter >= consecutive_count):
                        log_message(f"找到最接近目标J值：{min_J_bias_value}")
                        break

            #当前正确率偏度，需要增大填补正确率
            if cr_error < -max_cr_error:
                log_message("\n微调CR......")
                class_means = features_encoded.groupby(target).mean()
                # 找到所有错误填补的数据点
                error_positions = [
                    idx for idx in missing_indices
                    if filled_df.at[idx, 'city'] != clean_df.at[idx, 'city']
                ]

                if len(error_positions) == 0:
                    log_message("ERROR!")
                else:
                    # 计算每个错误数据点到其正确类别的距离
                    distances_to_correct = []
                    for idx in error_positions:
                        correct_class = clean_df.at[idx, 'city']
                        current_features = features_encoded.loc[idx]

                        # 检查正确类别是否在 filled_df 中出现过
                        if correct_class not in filled_df['city'].unique():
                            # 如果未出现过，直接将距离设置为 0
                            distance = 0
                            log_message(f"Class '{correct_class}' not found in filled_df. Setting distance to 0.")
                        else:
                            # 如果出现过，计算实际距离
                            correct_class_mean = class_means.loc[correct_class]
                            distance = np.linalg.norm(current_features - correct_class_mean)

                        distances_to_correct.append((idx, distance))

                    # 根据J_error的符号选择数据点
                    n_corrections = min(len(error_positions), int(abs(cr_error) * len(missing_indices)))
                    log_message(f"Number of corrections to make: {n_corrections}")

                    if J_error < 0:  # 选择距离最小的错误数据点
                        distances_to_correct.sort(key=lambda x: x[1])  # 按距离从小到大排序
                    else:  # 选择距离最大的错误数据点
                        distances_to_correct.sort(key=lambda x: x[1], reverse=True)  # 按距离从大到小排序

                    # 修改选定的数据点为正确类别
                    for idx, _ in distances_to_correct[:n_corrections]:
                        filled_df.at[idx, 'city'] = clean_df.at[idx, 'city']
                        log_message(f"Corrected index {idx} to the correct class {clean_df.at[idx, 'city']}")

            #当前正确率偏高，需要减小填补正确率
            elif cr_error > max_cr_error:
                log_message("\n微调CR......")
                class_means = features_encoded.groupby(target).mean()
                # 找到所有正确填补的数据点
                correct_positions = [
                    idx for idx in missing_indices
                    if filled_df.at[idx, 'city'] == clean_df.at[idx, 'city']
                ]
                if len(correct_positions) == 0:
                    log_message("No correct positions to corrupt. ERROR!")
                else:
                    # 计算每个正确数据点到其正确类别中心的距离
                    distances_to_corrupt = []
                    for idx in correct_positions:
                        correct_class = clean_df.at[idx, 'city']
                        current_features = features_encoded.loc[idx]
                        correct_class_mean = class_means.loc[correct_class]
                        distance = np.linalg.norm(current_features - correct_class_mean)
                        distances_to_corrupt.append((idx, distance))
                    # 计算需要调整的样本数量
                    n_corruptions = min(len(correct_positions), int(abs(cr_error) * len(missing_indices)))
                    log_message(f"Number of corruptions to make: {n_corruptions}")

                    # 根据 J_error 的符号选择数据点
                    if J_error < 0:  # 选择距离最大的正确数据点
                        distances_to_corrupt.sort(key=lambda x: x[1], reverse=True)  # 按距离从大到小排序
                    else:  # 选择距离最小的正确数据点
                        distances_to_corrupt.sort(key=lambda x: x[1])  # 按距离从小到大排序

                    # 修改选定的数据点为错误类别
                    for idx, _ in distances_to_corrupt[:n_corruptions]:
                        correct_class = clean_df.at[idx, 'city']
                        current_features = features_encoded.loc[idx]
                        # 找到距离最近/最远的错误类别
                        distances_to_other_classes = []
                        for class_name, class_mean in class_means.iterrows():
                            if class_name != correct_class:
                                distance = np.linalg.norm(current_features - class_mean)
                                distances_to_other_classes.append((class_name, distance))

                        if J_error < 0:  # 修改为距离最近的错误类别
                            distances_to_other_classes.sort(key=lambda x: x[1])
                        else:  # 修改为距离最远的错误类别
                            distances_to_other_classes.sort(key=lambda x: x[1], reverse=True)

                        # 选择目标类别
                        target_class = distances_to_other_classes[0][0]
                        filled_df.at[idx, 'city'] = target_class
                        log_message(f"Corrupted index {idx} from '{correct_class}' to '{target_class}'")

            #经过CR微调之后，需要进行重新计算
            if abs(cr_error) > max_cr_error:
                log_message("\n微调CR之后......")
                cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
                log_message(f"Now: J={J}, CR={cr}")
                cr_error = cr - cr_target
                J_error = J - j_target
                if abs(cr_error) <= max_cr_error:
                    if abs(J_error) < max_J_error:
                        min_J_bias_value = J
                        min_J_bias_cr = cr
                        min_J_bias_filled_df = filled_df.copy()
                        log_message("当前J值和CR都在可接受范围内！")
                        break
                    elif abs(J_error) < min_recent_J_bias:
                        min_recent_J_bias = abs(J_error)
                        min_J_bias_value = J
                        min_J_bias_cr = cr
                        min_J_bias_filled_df = filled_df
                        min_J_bias_counter = 0
                    else:
                        min_J_bias_counter = min_J_bias_counter + 1
                        if (min_J_bias_counter >= consecutive_count):
                            log_message(f"找到最接近目标J值：{min_J_bias_value}")
                            break

            # 微调 J 值的完整策略修改
            if abs(J_error) > max_J_error:
                log_message("\n微调J Value......")
                class_means = features_encoded.groupby(target).mean()
                pairwise_dist = pairwise_distances(class_means)
                np.fill_diagonal(pairwise_dist, np.inf)  # 屏蔽对角线

                # 动态计算调整幅度
                #adjustment_rate = min(0.8, 0.2 * (1 + abs(J_error) * 5))
                adjustment_rate = min(1, 0.5 * (1 + abs(J_error) * 5))

                if J_error < 0:  # 需要增大J值
                    log_message("Attempting to increase J value by reclassifying points to their nearest classes")

                    # 计算每个数据点到所有类中心的距离
                    distances_to_means = pairwise_distances(features_encoded, class_means)

                    # 找到每个数据点的最临近类别及其距离
                    nearest_classes = np.argmin(distances_to_means, axis=1)  # 最临近类别的索引
                    nearest_distances = np.min(distances_to_means, axis=1)  # 到最临近类别的距离

                    # 获取当前类别标签
                    current_classes = target.values

                    # 找到当前类别与最临近类别不同的点，并记录其索引和距离
                    reclassify_candidates = []
                    for idx in range(len(features_encoded)):
                        nearest_class_name = class_means.index[nearest_classes[idx]]  # 最临近类别的名称
                        current_class_name = current_classes[idx]

                        # 确保类别名称的比较是大小写一致的
                        if current_class_name.lower() != nearest_class_name.lower():
                            reclassify_candidates.append((idx, nearest_distances[idx]))

                    before_change = filled_df.copy()
                    # 如果没有符合条件的点，直接跳过
                    if len(reclassify_candidates) == 0:
                        log_message("No points need reclassification. All points are already in their nearest classes.")
                        break
                    else:
                        # 按距离从小到大排序，并选择距离最小的n个点
                        reclassify_candidates.sort(key=lambda x: x[1])  # 按距离排序
                        # 根据 reclassify_candidates 的长度动态调整随机数的范围
                        if len(reclassify_candidates) >= 10:
                            num_to_reclassify = random.randint(2, 10)
                        else:
                            num_to_reclassify = random.randint(2, len(reclassify_candidates))
                        # 分离正确分类和错误分类的点
                        correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                    filled_df.at[idx, 'city'] == clean_df.at[idx, 'city']]
                        incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                      filled_df.at[idx, 'city'] != clean_df.at[idx, 'city']]

                        # 确保至少选择一个正确分类的点和一个错误分类的点
                        if correct_class_candidates:
                            correct_idx, _ = correct_class_candidates[0]  # 距离最小的正确分类点
                        else:
                            correct_idx = None

                        if incorrect_class_candidates:
                            incorrect_idx, _ = incorrect_class_candidates[0]  # 距离最小的错误分类点
                        else:
                            incorrect_idx = None

                        # 如果没有正确分类或错误分类的点，直接选择前 num_to_reclassify 个点
                        if correct_idx is None or incorrect_idx is None:
                            reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
                        else:
                            # 合并这两个点
                            selected_indices = [correct_idx, incorrect_idx]
                            # 确保选择的点数不超过 num_to_reclassify
                            remaining_candidates = [idx for idx, _ in reclassify_candidates if
                                                    idx not in selected_indices]
                            additional_indices = remaining_candidates[:num_to_reclassify - 2]
                            selected_indices.extend(additional_indices)
                            reclassify_indices = selected_indices[:num_to_reclassify]

                        log_message(f"Reclassifying {len(reclassify_indices)} points to their nearest classes")
                        for idx in reclassify_indices:
                            current_class_name = current_classes[idx]
                            nearest_class_name = class_means.index[nearest_classes[idx]]  # 最临近类别的名称

                            # 更新类别
                            filled_df.at[idx, 'city'] = nearest_class_name
                            log_message(f"Reclassified index {idx} from '{current_class_name}' to '{nearest_class_name}'")

                if J_error > 0:  # 需要减小J值
                    log_message("Attempting to decrease J value by reclassifying points to the farthest non-same classes")
                    # 计算每个数据点到所有类中心的距离
                    distances_to_means = pairwise_distances(features_encoded, class_means)
                    # 找到每个数据点的最临近类别及其距离
                    nearest_classes = np.argmin(distances_to_means, axis=1)  # 最临近类别的索引
                    nearest_distances = np.min(distances_to_means, axis=1)  # 到最临近类别的距离
                    # 获取当前类别标签
                    current_classes = target.values
                    # 找到当前类别与最临近类别相同的点，并记录其索引和距离
                    reclassify_candidates = []
                    for idx in range(len(features_encoded)):
                        nearest_class_name = class_means.index[nearest_classes[idx]]  # 最临近类别的名称
                        current_class_name = current_classes[idx]
                        # 确保类别名称的比较是大小写一致的
                        if current_class_name.lower() == nearest_class_name.lower():
                            # 计算该数据点到所有非相同类别的距离
                            distances_to_other_classes = []
                            for class_name, class_mean in class_means.iterrows():
                                if class_name != current_class_name:
                                    distance = np.linalg.norm(features_encoded.iloc[idx] - class_mean)
                                    distances_to_other_classes.append((class_name, distance))
                            # 选择距离最远的非相同类别
                            if distances_to_other_classes:
                                farthest_class = max(distances_to_other_classes, key=lambda x: x[1])[0]
                                reclassify_candidates.append((idx, farthest_class))
                    # 如果没有符合条件的点，直接跳过
                    if len(reclassify_candidates) == 0:
                        log_message("No points need reclassification. All points are already in their farthest non-same classes.")
                        break
                    else:
                        # 按距离从大到小排序，并选择距离最大的n个点
                        reclassify_candidates.sort(key=lambda x: x[1], reverse=True)  # 按距离排序
                        # 根据 reclassify_candidates 的长度动态调整随机数的范围
                        if len(reclassify_candidates) >= 10:
                            num_to_reclassify = random.randint(2, 10)
                        else:
                            num_to_reclassify = random.randint(2, len(reclassify_candidates))
                        # 分离正确分类和错误分类的点
                        correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                    filled_df.at[idx, 'city'] == clean_df.at[idx, 'city']]
                        incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                      filled_df.at[idx, 'city'] != clean_df.at[idx, 'city']]

                        # 确保至少选择一个正确分类的点和一个错误分类的点
                        if correct_class_candidates:
                            correct_idx, _ = correct_class_candidates[0]  # 距离最大的正确分类点
                        else:
                            correct_idx = None

                        if incorrect_class_candidates:
                            incorrect_idx, _ = incorrect_class_candidates[0]  # 距离最大的错误分类点
                        else:
                            incorrect_idx = None

                        # 如果没有正确分类或错误分类的点，直接选择前 num_to_reclassify 个点
                        if correct_idx is None or incorrect_idx is None:
                            reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
                        else:
                            # 合并这两个点
                            selected_indices = [correct_idx, incorrect_idx]
                            # 确保选择的点数不超过 num_to_reclassify
                            remaining_candidates = [idx for idx, _ in reclassify_candidates if
                                                    idx not in selected_indices]
                            additional_indices = remaining_candidates[:num_to_reclassify - 2]
                            selected_indices.extend(additional_indices)
                            reclassify_indices = selected_indices[:num_to_reclassify]

                        log_message(f"Reclassifying {len(reclassify_indices)} points to their farthest non-same classes")

                        for idx in reclassify_indices:
                            current_class_name = current_classes[idx]
                            farthest_class_name = reclassify_candidates[reclassify_indices.index(idx)][1]
                            # 更新类别
                            filled_df.at[idx, 'city'] = farthest_class_name
                            log_message(f"Reclassified index {idx} from '{current_class_name}' to '{farthest_class_name}'")

            current_iter += 1

        if(j_target <= max_J_value):
            # 保存结果
            filled_file = os.path.join(save_path, f'filled-j-{min_J_bias_value:.3f}-cr-{min_J_bias_cr:.3f}.csv')
            min_J_bias_filled_df.to_csv(filled_file, index=False)
            log_message(f"{min_J_bias_filled_df}已保存到{save_path}")
            save_results(cr_target,j_target,min_J_bias_cr,min_J_bias_value)

# 保存实验结果
#pd.DataFrame(results).to_csv(os.path.join(save_path, 'results.csv'), index=False)
print("实验结果已保存到 results.csv")