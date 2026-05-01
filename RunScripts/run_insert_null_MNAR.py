# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import re


def insert_null_Beers(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于城市名称长度（长名称更容易缺失） ==========
    if 'city' in df.columns:
        # 计算城市名称长度
        city_length = df['city'].astype(str).str.len()

        # 长名称城市更容易缺失
        length_75 = city_length.quantile(0.75)
        long_name_mask = city_length > length_75
        missing_score[long_name_mask] += 3.0

        # 短名称城市不容易缺失
        length_25 = city_length.quantile(0.25)
        short_name_mask = city_length < length_25
        missing_score[short_name_mask] -= 0.5

    # ========== 3. 基于城市名称中包含特殊字符 ==========
    if 'city' in df.columns:
        # 包含连字符、空格等的城市更容易缺失
        has_hyphen = df['city'].astype(str).str.contains('-', na=False)
        missing_score[has_hyphen] += 1.5

        has_space = df['city'].astype(str).str.contains(' ', na=False)
        missing_score[has_space] += 0.5

    # ========== 4. 基于城市名称的罕见程度（自动识别罕见城市） ==========
    if 'city' in df.columns:
        # 计算每个城市的出现频率
        city_counts = df['city'].value_counts()
        city_freq = city_counts / len(df)

        # 罕见城市（出现频率低于2%）更容易缺失
        for city in city_counts.index:
            if city_freq[city] < 0.1:  # 出现率低于10%
                mask = df['city'] == city
                missing_score[mask] += 3.0
            elif city_freq[city] < 0.05:  # 出现率低于5%
                mask = df['city'] == city
                missing_score[mask] += 1.5

    # ========== 8. 基于州与城市的关联 ==========
    if 'state' in df.columns and 'city' in df.columns:
        # 特定州的城市更容易缺失
        state_counts = df['state'].value_counts()
        state_freq = state_counts / len(df)

        # 出现频率低的州，其城市更容易缺失
        for state in state_counts.index:
            if state_freq[state] < 0.05:
                mask = df['state'] == state
                missing_score[mask] += 2.0
            elif state_freq[state] < 0.10:
                mask = df['state'] == state
                missing_score[mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声，使缺失更依赖于城市本身特征
    random_noise = np.random.uniform(0, 0.3, n)  # 噪声更小
    final_prob = missing_score * 0.7 + random_noise * 0.3  # 更依赖城市特征
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Flights(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于航班号长度 ==========
    if 'flight' in df.columns:
        flight_length = df['flight'].astype(str).str.len()

        # 长航班号更容易缺失
        length_75 = flight_length.quantile(0.75)
        long_flight_mask = flight_length > length_75
        missing_score[long_flight_mask] += 3.0

        # 短航班号不容易缺失
        length_25 = flight_length.quantile(0.25)
        short_flight_mask = flight_length < length_25
        missing_score[short_flight_mask] -= 0.5


    # ========== 3. 基于航班号前缀（航空公司代码） ==========
    if 'flight' in df.columns:
        flight_str = df['flight'].astype(str)

        # 提取前缀（字母部分）
        prefix = flight_str.str.extract(r'^([A-Za-z]+)')[0]

        if prefix.notna().any():
            # 计算每个前缀的出现频率
            prefix_counts = prefix.value_counts()
            prefix_freq = prefix_counts / len(df)

            # 罕见前缀更容易缺失
            for p in prefix_counts.index:
                if prefix_freq[p] < 0.1:  # 出现率低于10%
                    mask = prefix == p
                    missing_score[mask] += 3.0
                elif prefix_freq[p] < 0.05:  # 出现率低于5%
                    mask = prefix == p
                    missing_score[mask] += 1.5

    # ========== 5. 基于航班号的罕见程度 ==========
    if 'flight' in df.columns:
        flight_counts = df['flight'].value_counts()
        flight_freq = flight_counts / len(df)

        # 罕见航班号（出现频率低于0.5%）更容易缺失
        for flight in flight_counts.index:
            if flight_freq[flight] < 0.1:
                mask = df['flight'] == flight
                missing_score[mask] += 3.0
            elif flight_freq[flight] < 0.05:
                mask = df['flight'] == flight
                missing_score[mask] += 2.0
            elif flight_freq[flight] < 0.02:
                mask = df['flight'] == flight
                missing_score[mask] += 1.0

    # ========== 7. 基于航班号与航空公司的关联 ==========
    if 'src' in df.columns and 'flight' in df.columns:
        # 特定航空公司的航班号更容易缺失
        src_counts = df['src'].value_counts()
        src_freq = src_counts / len(df)

        for src in src_counts.index:
            if src_freq[src] < 0.05:  # 小型航空公司
                mask = df['src'] == src
                missing_score[mask] += 2.0
            elif src_freq[src] < 0.10:  # 中型航空公司
                mask = df['src'] == src
                missing_score[mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声，使缺失更依赖于航班号本身特征
    random_noise = np.random.uniform(0, 0.3, n)  # 噪声更小
    final_prob = missing_score * 0.7 + random_noise * 0.3  # 更依赖航班号特征
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Hospital(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于城市名称长度 ==========
    if 'City' in df.columns:
        city_length = df['City'].astype(str).str.len()

        # 长名称城市更容易缺失
        length_75 = city_length.quantile(0.75)
        long_name_mask = city_length > length_75
        missing_score[long_name_mask] += 3.0

        # 短名称城市不容易缺失
        length_25 = city_length.quantile(0.25)
        short_name_mask = city_length < length_25
        missing_score[short_name_mask] -= 0.5

    # ========== 2. 基于城市名称中的特殊字符 ==========
    if 'City' in df.columns:
        city_str = df['City'].astype(str)

        # 包含空格的城市更容易缺失
        has_space = city_str.str.contains(' ', na=False)
        missing_score[has_space] += 1.5

        # 包含连字符的城市更容易缺失
        has_hyphen = city_str.str.contains('-', na=False)
        missing_score[has_hyphen] += 2.0

        # 包含数字的城市更容易缺失
        has_digit = city_str.str.contains(r'\d', na=False)
        missing_score[has_digit] += 2.5

        # 包含句点的城市更容易缺失
        has_dot = city_str.str.contains('\.', na=False)
        missing_score[has_dot] += 2.0

    # ========== 5. 基于城市的罕见程度（出现频率） ==========
    if 'City' in df.columns:
        city_counts = df['City'].value_counts()
        city_freq = city_counts / len(df)

        # 罕见城市更容易缺失
        for city in city_counts.index:
            if city_freq[city] < 0.1:  # 出现率低于10%
                mask = df['City'] == city
                missing_score[mask] += 3.0
            elif city_freq[city] < 0.05:  # 出现率低于5%
                mask = df['City'] == city
                missing_score[mask] += 2.0
            elif city_freq[city] < 0.02:  # 出现率低于2%
                mask = df['City'] == city
                missing_score[mask] += 1.0

    # ========== 7. 基于城市名称与州的关联 ==========
    if 'State' in df.columns and 'City' in df.columns:
        # 特定州的城市更容易缺失
        state_counts = df['State'].value_counts()
        state_freq = state_counts / len(df)

        # 出现频率低的州，其城市更容易缺失
        for state in state_counts.index:
            if state_freq[state] < 0.1:  # 出现率低于10%
                mask = df['State'] == state
                missing_score[mask] += 2.0
            elif state_freq[state] < 0.05:  # 出现率低于5%
                mask = df['State'] == state
                missing_score[mask] += 1.0

    # ========== 8. 基于城市名称的长度模式 ==========
    if 'City' in df.columns:
        city_length = df['City'].astype(str).str.len()

        # 极长城市名称（>15字符）更容易缺失
        very_long_mask = city_length > 15
        missing_score[very_long_mask] += 2.0

        # 极短城市名称（<3字符）更容易缺失
        very_short_mask = city_length < 3
        missing_score[very_short_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声，使缺失更依赖于城市本身特征
    random_noise = np.random.uniform(0, 0.3, n)  # 噪声更小
    final_prob = missing_score * 0.7 + random_noise * 0.3  # 更依赖城市特征
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )
    df.loc[missing_indices, target_column] = pd.NA
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_RedWineQuality(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于质量分数本身（主要依赖） ==========
    if 'quality' in df.columns:
        quality = df['quality'].values

        # 极端质量分数更容易缺失
        # 最低质量（3分）更容易缺失
        very_low_mask = quality == 3
        missing_score[very_low_mask] += 2.0

        # 次低质量（4分）容易缺失
        low_mask = quality == 4
        missing_score[low_mask] += 1.0

        # 中等偏低（5分）一般
        medium_low_mask = quality == 5
        missing_score[medium_low_mask] += 0.0

        # 中等（6分）不容易缺失
        medium_mask = quality == 6
        missing_score[medium_mask] += 0.0

        # 中等偏高（7分）一般
        medium_high_mask = quality == 7
        missing_score[medium_high_mask] += 1.0

        # 最高质量（8分）容易缺失
        high_mask = quality == 8
        missing_score[high_mask] += 2.0

    # ========== 2. 基于质量分数的稀有性 ==========
    if 'quality' in df.columns:
        quality_counts = df['quality'].value_counts()
        quality_freq = quality_counts / len(df)

        # 罕见质量分数更容易缺失
        for q in quality_counts.index:
            if quality_freq[q] < 0.1:  # 出现率低于10%
                mask = df['quality'] == q
                missing_score[mask] += 3.0
            elif quality_freq[q] < 0.05:  # 出现率低于5%
                mask = df['quality'] == q
                missing_score[mask] += 1.0


    # ========== 4. 基于质量分数与总体的关系 ==========
    if 'quality' in df.columns:
        # 质量分数排名
        quality_rank = df['quality'].rank(pct=True)

        # 排名前5%（8分为主）更容易缺失
        top_5_mask = quality_rank > 0.95
        missing_score[top_5_mask] += 1.5

        # 排名后5%（3分为主）更容易缺失
        bottom_5_mask = quality_rank < 0.05
        missing_score[bottom_5_mask] += 2.0

    # ========== 5. 基于质量分数的边界值 ==========
    if 'quality' in df.columns:
        # 边界值（3分和8分）更容易缺失
        is_boundary = (df['quality'] == 3) | (df['quality'] == 8)
        missing_score[is_boundary] += 1.5

    # ========== 6. 基于质量分数与标准差的偏离 ==========
    if 'quality' in df.columns:
        quality_mean = df['quality'].mean()
        quality_std = df['quality'].std()

        # 偏离均值越远，越容易缺失
        deviation = np.abs(df['quality'] - quality_mean)
        deviation_normalized = deviation / quality_std if quality_std > 0 else deviation

        # 偏差超过1.5倍标准差的更容易缺失
        high_deviation_mask = deviation_normalized > 1.5
        missing_score[high_deviation_mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声，使缺失更依赖于质量本身特征
    random_noise = np.random.uniform(0, 0.3, n)  # 噪声更小
    final_prob = missing_score * 0.7 + random_noise * 0.3  # 更依赖质量特征
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )
    df.loc[missing_indices, target_column] = pd.NA
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_AvocadoRipeness(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于成熟度类别本身（主要依赖） ==========
    if 'ripeness' in df.columns:
        ripeness = df['ripeness'].values

        # hard（未熟）更容易缺失
        hard_mask = ripeness == 'hard'
        missing_score[hard_mask] += 1.0

        # breaking（开始成熟）容易缺失
        breaking_mask = ripeness == 'breaking'
        missing_score[breaking_mask] += 1.0

        # firm-ripe（半熟）一般容易缺失
        firm_ripe_mask = ripeness == 'firm-ripe'
        missing_score[firm_ripe_mask] += 0.5

        # ripe（完全成熟）最不容易缺失
        ripe_mask = ripeness == 'ripe'
        missing_score[ripe_mask] += 0.0

        # pre-conditioned（预处理）容易缺失
        pre_conditioned_mask = ripeness == 'pre-conditioned'
        missing_score[pre_conditioned_mask] += 0.0

    # ========== 2. 基于成熟度的稀有性 ==========
    if 'ripeness' in df.columns:
        ripeness_counts = df['ripeness'].value_counts()
        ripeness_freq = ripeness_counts / len(df)

        # 罕见成熟度更容易缺失
        for r in ripeness_counts.index:
            if ripeness_freq[r] < 0.05:  # 出现率低于5%
                mask = df['ripeness'] == r
                missing_score[mask] += 2.0
            elif ripeness_freq[r] < 0.10:  # 出现率低于10%
                mask = df['ripeness'] == r
                missing_score[mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声，使缺失更依赖于成熟度本身特征
    random_noise = np.random.uniform(0, 0.3, n)  # 噪声更小
    final_prob = missing_score * 0.7 + random_noise * 0.3  # 更依赖成熟度特征
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_numericaldatasets(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于目标本身（主要依赖） ==========
    if target_column in df.columns:
        strength = df[target_column].values

        # 强度极低（<25分位数）更容易缺失
        str_25 = df[target_column].quantile(0.25)
        very_low_mask = strength < str_25
        missing_score[very_low_mask] += 2.0

        # 强度极高（>75分位数）更容易缺失
        str_75 = df[target_column].quantile(0.75)
        very_high_mask = strength > str_75
        missing_score[very_high_mask] += 2.0

        # 边界值（最小或最大）更容易缺失
        is_min = strength == strength.min()
        is_max = strength == strength.max()
        missing_score[is_min | is_max] += 2.5

    # ========== 2. 基于目标的稀有性 ==========
    if target_column in df.columns:
        strength_counts = df[target_column].value_counts()
        strength_freq = strength_counts / len(df)

        # 罕见强度（出现频率低于10%）更容易缺失
        for s in strength_counts.index:
            if strength_freq[s] < 0.1:
                mask = df[target_column] == s
                missing_score[mask] += 2.5
            elif strength_freq[s] < 0.05:
                mask = df[target_column] == s
                missing_score[mask] += 1.0

    # ========== 4. 基于目标与均值的偏离 ==========
    if target_column in df.columns:
        strength_mean = df[target_column].mean()
        strength_std = df[target_column].std()

        deviation = np.abs(df[target_column] - strength_mean)
        deviation_normalized = deviation / strength_std if strength_std > 0 else deviation

        # 偏离均值超过2倍标准差更容易缺失
        high_deviation_mask = deviation_normalized > 2
        missing_score[high_deviation_mask] += 2.5

        # 偏离均值超过1.5倍标准差容易缺失
        medium_deviation_mask = (deviation_normalized > 1.5) & (deviation_normalized <= 2)
        missing_score[medium_deviation_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # MNAR：添加更小的随机噪声
    random_noise = np.random.uniform(0, 0.3, n)
    final_prob = missing_score * 0.7 + random_noise * 0.3
    final_prob = final_prob / final_prob.sum()

    # 选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )
    df.loc[missing_indices, target_column] = pd.NA
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')








datasets = {
    #"ETTh1": "OT",
    #"ETTm1": "OT",
    #"Illness": "OT",
    #"Exchange": "OT",
    "Weather": "OT",

    #"Beers": "city",
    #"Flights": "flight",
    #"Hospital": "City",
    #"RedWineQuality": "quality",
    #"AvocadoRipeness": "ripeness",

    #"concrete": "concrete_compressive_strength",
    #"CCPP": "PE",
    #"AirfoilSelfNoise": "SSPL",
    #"Abalone": "Rings",
    #"ParisHousing": "price"
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
base_path = "../Datasets"
for dataset, target_column in datasets.items():
    input_file = os.path.join(base_path, dataset, "clean.csv")
    output_path = os.path.join(base_path, dataset, "null", "MNAR")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        output_file = os.path.join(output_path, f'dirty-{rate}.csv')
        if dataset == "Beers":
            insert_null_Beers(input_file, output_file, rate, target_column)
        elif dataset == "Flights":
            insert_null_Flights(input_file, output_file, rate, target_column)
        elif dataset == "Hospital":
            insert_null_Hospital(input_file, output_file, rate, target_column)
        elif dataset == "RedWineQuality":
            insert_null_RedWineQuality(input_file, output_file, rate, target_column)
        elif dataset == "AvocadoRipeness":
            insert_null_AvocadoRipeness(input_file, output_file, rate, target_column)

        elif dataset == "concrete" or dataset == "CCPP" or dataset == "AirfoilSelfNoise" or dataset == "Abalone" or dataset == "ParisHousing":
            insert_null_numericaldatasets(input_file, output_file, rate, target_column)

        elif dataset == "ETTh1" or dataset == "ETTm1" or dataset == "Illness" or dataset == "Exchange" or dataset == "Weather":
            insert_null_numericaldatasets(input_file, output_file, rate, target_column)
