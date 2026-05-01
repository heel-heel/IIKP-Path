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

    # ========== 1. 基于啤酒风格（自动识别罕见风格） ==========
    if 'style' in df.columns:
        # 计算每种风格的出现频率
        style_counts = df['style'].value_counts()
        style_freq = style_counts / len(df)

        # 罕见风格（出现频率低于10%）更容易缺失
        for style in style_counts.index:
            if style_freq[style] < 0.1:  # 出现率低于10%视为罕见风格
                mask = df['style'] == style
                missing_score[mask] += 2.0
            elif style_freq[style] < 0.2:  # 出现率低于20%视为中等罕见
                mask = df['style'] == style
                missing_score[mask] += 1.0

    # ========== 2. 基于酒精度（自动识别高酒精度） ==========
    if 'abv' in df.columns:
        # 自动计算75分位数作为高酒精度阈值
        abv_75 = df['abv'].quantile(0.75)
        abv_90 = df['abv'].quantile(0.90)

        # 高酒精度（高于75分位数）更容易缺失
        high_abv_mask = df['abv'] > abv_75
        missing_score[high_abv_mask] += 1.5

        # 极高酒精度（高于90分位数）更容易缺失
        very_high_abv_mask = df['abv'] > abv_90
        missing_score[very_high_abv_mask] += 1.0

        # 低酒精度（低于25分位数）不容易缺失
        abv_25 = df['abv'].quantile(0.25)
        low_abv_mask = df['abv'] < abv_25
        missing_score[low_abv_mask] -= 0.5

    # ========== 3. 基于苦味值（自动识别高IBU） ==========
    if 'ibu' in df.columns:
        # 自动计算75分位数作为高IBU阈值
        ibu_75 = df['ibu'].quantile(0.75)
        ibu_90 = df['ibu'].quantile(0.90)

        # 高IBU（高于75分位数）更容易缺失
        high_ibu_mask = df['ibu'] > ibu_75
        missing_score[high_ibu_mask] += 1.5

        # 极高IBU（高于90分位数）更容易缺失
        very_high_ibu_mask = df['ibu'] > ibu_90
        missing_score[very_high_ibu_mask] += 1.0

        # 低IBU（低于25分位数）不容易缺失
        ibu_25 = df['ibu'].quantile(0.25)
        low_ibu_mask = df['ibu'] < ibu_25
        missing_score[low_ibu_mask] -= 0.5

    # ========== 4. 基于酒厂规模（自动识别小型酒厂） ==========
    if 'brewery_id' in df.columns:
        # 计算每个酒厂的出现次数（规模）
        brewery_counts = df.groupby('brewery_id').size()
        df['_temp_brewery_size'] = df['brewery_id'].map(brewery_counts)

        # 自动计算小型酒厂阈值（出现次数低于25分位数）
        size_25 = df['_temp_brewery_size'].quantile(0.25)
        size_10 = df['_temp_brewery_size'].quantile(0.10)

        # 小型酒厂（出现次数低于25分位数）更容易缺失
        small_brewery_mask = df['_temp_brewery_size'] <= size_25
        missing_score[small_brewery_mask] += 2.0

        # 微型酒厂（出现次数低于10分位数）更容易缺失
        micro_brewery_mask = df['_temp_brewery_size'] <= size_10
        missing_score[micro_brewery_mask] += 1.5

    # ========== 归一化缺失概率分数 ==========
    # 确保分数非负
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    # 归一化到 [0, 1]
    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n  # 如果全为0，均匀分布

    # 添加随机噪声，避免完全确定性
    random_noise = np.random.uniform(0, 0.3, n)
    final_prob = missing_score * 0.7 + random_noise * 0.3
    final_prob = final_prob / final_prob.sum()

    # 根据概率选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )

    df.loc[missing_indices, target_column] = pd.NA

    # 清理临时列
    if '_temp_brewery_size' in df.columns:
        df = df.drop(columns=['_temp_brewery_size'])

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Flights(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    def convert_time_to_24h(time_str):
        if pd.isna(time_str):
            return np.nan
        time_str = str(time_str).strip().lower()

        # 提取数字和上午/下午标识
        match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', time_str)
        if not match:
            return np.nan

        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3) if match.group(3) else None

        # 转换为24小时制
        if period == 'pm' or period == 'p.m.':
            if hour != 12:
                hour += 12
        elif period == 'am' or period == 'a.m.':
            if hour == 12:
                hour = 0

        # 返回整数格式（如 1455）
        return hour * 100 + minute

    # ========== 1. 基于航空公司（自动识别罕见航空公司） ==========
    if 'src' in df.columns:
        # 计算每个航空公司的出现频率
        src_counts = df['src'].value_counts()
        src_freq = src_counts / len(df)

        # 罕见航空公司（出现频率低于5%）更容易缺失
        for src in src_counts.index:
            if src_freq[src] < 0.05:  # 出现率低于5%
                mask = df['src'] == src
                missing_score[mask] += 3.0
            elif src_freq[src] < 0.10:  # 出现率低于10%
                mask = df['src'] == src
                missing_score[mask] += 1.5
            elif src_freq[src] < 0.20:  # 出现率低于20%
                mask = df['src'] == src
                missing_score[mask] += 0.5

    # ========== 2. 基于计划起飞时间（自动识别凌晨/深夜航班） ==========
    if 'sched_dep_time' in df.columns:
        # 将时间转换为小时数
        sched_hour = df['sched_dep_time'].apply(convert_time_to_24h)// 100

        # 凌晨航班（0-5点）更容易缺失
        night_mask = (sched_hour >= 0) & (sched_hour < 6)
        missing_score[night_mask] += 2.0

        # 深夜航班（22-24点）更容易缺失
        late_mask = (sched_hour >= 22) & (sched_hour <= 24)
        missing_score[late_mask] += 1.5

        # 高峰时段（7-9点，17-19点）不容易缺失
        peak_mask = ((sched_hour >= 7) & (sched_hour <= 9)) | ((sched_hour >= 17) & (sched_hour <= 19))
        missing_score[peak_mask] -= 0.5

    # ========== 3. 基于延误情况（自动识别延误严重的航班） ==========
    if 'sched_dep_time' in df.columns and 'act_dep_time' in df.columns:
        # 计算实际与计划的差值（延误时长）
        # 处理缺失的实际起飞时间
        act_dep_valid = df['act_dep_time'].fillna(df['sched_dep_time'])
        delay = act_dep_valid.apply(convert_time_to_24h) - df['sched_dep_time'].apply(convert_time_to_24h)

        # 延误越严重，缺失概率越高
        if len(delay) > 0:
            # 自动计算75分位数作为严重延误阈值
            delay_75 = delay.quantile(0.75)
            delay_90 = delay.quantile(0.90)

            # 严重延误（>75分位数）更容易缺失
            severe_delay_mask = delay > delay_75
            missing_score[severe_delay_mask] += 2.0

            # 极度延误（>90分位数）更容易缺失
            extreme_delay_mask = delay > delay_90
            missing_score[extreme_delay_mask] += 1.5

            # 准点或提前到达不容易缺失
            ontime_mask = delay <= 0
            missing_score[ontime_mask] -= 0.5

    # ========== 4. 基于实际起飞时间缺失情况 ==========
    if 'act_dep_time' in df.columns:
        # 实际起飞时间缺失的航班，航班号也更容易缺失
        act_dep_missing_mask = df['act_dep_time'].isna()
        missing_score[act_dep_missing_mask] += 2.5

    # ========== 5. 基于实际到达时间缺失情况 ==========
    if 'act_arr_time' in df.columns:
        # 实际到达时间缺失的航班，航班号也更容易缺失
        act_arr_missing_mask = df['act_arr_time'].isna()
        missing_score[act_arr_missing_mask] += 2.0

    # ========== 归一化缺失概率分数 ==========
    # 确保分数非负
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    # 归一化到 [0, 1]
    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声，避免完全确定性
    random_noise = np.random.uniform(0, 0.3, n)
    final_prob = missing_score * 0.7 + random_noise * 0.3
    final_prob = final_prob / final_prob.sum()

    # 根据概率选择缺失样本
    missing_indices = np.random.choice(
        n,
        size=n_missing,
        replace=False,
        p=final_prob
    )
    df.loc[missing_indices, 'flight'] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Hospital(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于州（自动识别偏远/罕见州） ==========
    if 'State' in df.columns:
        # 计算每个州的出现频率
        state_counts = df['State'].value_counts()
        state_freq = state_counts / len(df)

        # 罕见州（出现频率低于5%）更容易缺失
        for state in state_counts.index:
            if state_freq[state] < 0.05:
                mask = df['State'] == state
                missing_score[mask] += 3.0
            elif state_freq[state] < 0.10:
                mask = df['State'] == state
                missing_score[mask] += 1.5
            elif state_freq[state] < 0.20:
                mask = df['State'] == state
                missing_score[mask] += 0.5

    # ========== 2. 基于医院类型 ==========
    if 'HospitalType' in df.columns:
        # 自动识别罕见医院类型
        type_counts = df['HospitalType'].value_counts()
        type_freq = type_counts / len(df)

        for htype in type_counts.index:
            if type_freq[htype] < 0.05:
                mask = df['HospitalType'] == htype
                missing_score[mask] += 2.5
            elif type_freq[htype] < 0.15:
                mask = df['HospitalType'] == htype
                missing_score[mask] += 1.0

    # ========== 3. 基于医院所有者 ==========
    if 'HospitalOwner' in df.columns:
        # 自动识别罕见所有者类型
        owner_counts = df['HospitalOwner'].value_counts()
        owner_freq = owner_counts / len(df)

        for owner in owner_counts.index:
            if owner_freq[owner] < 0.05:
                mask = df['HospitalOwner'] == owner
                missing_score[mask] += 2.0
            elif owner_freq[owner] < 0.15:
                mask = df['HospitalOwner'] == owner
                missing_score[mask] += 0.8

    # ========== 4. 基于急诊服务 ==========
    if 'EmergencyService' in df.columns:
        # 没有急诊服务的医院更容易缺失城市信息
        no_emergency_mask = df['EmergencyService'] == 'No'
        missing_score[no_emergency_mask] += 2.0

    # ========== 5. 基于医院名称长度 ==========
    if 'HospitalName' in df.columns:
        # 医院名称长度（长名称可能信息不全）
        name_length = df['HospitalName'].astype(str).str.len()
        name_75 = name_length.quantile(0.75)

        long_name_mask = name_length > name_75
        missing_score[long_name_mask] += 1.5

        short_name_mask = name_length < 20
        missing_score[short_name_mask] -= 0.5

    # ========== 6. 基于电话号码存在性 ==========
    if 'PhoneNumber' in df.columns:
        # 电话号码缺失或异常的医院更容易缺失城市信息
        phone_missing_mask = df['PhoneNumber'].isna()
        missing_score[phone_missing_mask] += 2.5

        # 电话号码格式异常（长度不对）
        phone_length = df['PhoneNumber'].astype(str).str.len()
        abnormal_phone_mask = (phone_length < 10) | (phone_length > 12)
        missing_score[abnormal_phone_mask] += 1.0

    # ========== 7. 基于地址信息完整性 ==========
    if 'Address1' in df.columns:
        # 地址1缺失的医院更容易缺失城市信息
        addr1_missing_mask = df['Address1'].isna()
        missing_score[addr1_missing_mask] += 2.0

    # ========== 8. 基于县名存在性 ==========
    if 'CountyName' in df.columns:
        # 县名缺失的医院更容易缺失城市信息
        county_missing_mask = df['CountyName'].isna()
        missing_score[county_missing_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_RedWineQuality(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于酒精含量 ==========
    if 'alcohol' in df.columns:
        # 酒精含量越低，质量数据越容易缺失
        alcohol_25 = df['alcohol'].quantile(0.25)
        alcohol_75 = df['alcohol'].quantile(0.75)

        low_alcohol_mask = df['alcohol'] < alcohol_25
        missing_score[low_alcohol_mask] += 2.0

        high_alcohol_mask = df['alcohol'] > alcohol_75
        missing_score[high_alcohol_mask] -= 0.5  # 高酒精度更完整

    # ========== 2. 基于挥发性酸度 ==========
    if 'volatile_acidity' in df.columns:
        # 挥发性酸度过高（酒变质）更容易缺失质量数据
        va_75 = df['volatile_acidity'].quantile(0.75)
        va_90 = df['volatile_acidity'].quantile(0.90)

        high_va_mask = df['volatile_acidity'] > va_75
        missing_score[high_va_mask] += 1.5

        very_high_va_mask = df['volatile_acidity'] > va_90
        missing_score[very_high_va_mask] += 1.0

    # ========== 3. 基于固定酸度 ==========
    if 'fixed_acidity' in df.columns:
        # 酸度过高或过低都容易缺失
        fa_25 = df['fixed_acidity'].quantile(0.25)
        fa_75 = df['fixed_acidity'].quantile(0.75)

        low_acid_mask = df['fixed_acidity'] < fa_25
        missing_score[low_acid_mask] += 1.0

        high_acid_mask = df['fixed_acidity'] > fa_75
        missing_score[high_acid_mask] += 1.0

    # ========== 4. 基于密度 ==========
    if 'density' in df.columns:
        # 密度异常的酒更容易缺失质量数据
        density_25 = df['density'].quantile(0.25)
        density_75 = df['density'].quantile(0.75)

        low_density_mask = df['density'] < density_25
        missing_score[low_density_mask] += 1.0

        high_density_mask = df['density'] > density_75
        missing_score[high_density_mask] += 1.0

    # ========== 5. 基于总二氧化硫 ==========
    if 'total_sulfur_dioxide' in df.columns:
        # 二氧化硫含量过高或过低可能表示防腐剂使用异常
        so2_75 = df['total_sulfur_dioxide'].quantile(0.75)
        so2_90 = df['total_sulfur_dioxide'].quantile(0.90)

        high_so2_mask = df['total_sulfur_dioxide'] > so2_75
        missing_score[high_so2_mask] += 1.0

        very_high_so2_mask = df['total_sulfur_dioxide'] > so2_90
        missing_score[very_high_so2_mask] += 1.0

    # ========== 6. 基于pH值 ==========
    if 'pH' in df.columns:
        # 极端pH值更容易缺失
        ph_25 = df['pH'].quantile(0.25)
        ph_75 = df['pH'].quantile(0.75)

        low_ph_mask = df['pH'] < ph_25
        missing_score[low_ph_mask] += 1.0

        high_ph_mask = df['pH'] > ph_75
        missing_score[high_ph_mask] += 1.0

    # ========== 7. 基于氯含量 ==========
    if 'chlorides' in df.columns:
        # 氯含量过高（可能影响口感）更容易缺失
        chlorides_75 = df['chlorides'].quantile(0.75)
        high_chlorides_mask = df['chlorides'] > chlorides_75
        missing_score[high_chlorides_mask] += 1.0

    # ========== 8. 基于残糖 ==========
    if 'residual_sugar' in df.columns:
        # 残糖异常（甜酒/干酒边界）更容易缺失
        sugar_75 = df['residual_sugar'].quantile(0.75)
        high_sugar_mask = df['residual_sugar'] > sugar_75
        missing_score[high_sugar_mask] += 0.5

    # ========== 9. 基于柠檬酸 ==========
    if 'citric_acid' in df.columns:
        # 柠檬酸含量异常
        citric_75 = df['citric_acid'].quantile(0.75)
        high_citric_mask = df['citric_acid'] > citric_75
        missing_score[high_citric_mask] += 0.5

    # ========== 10. 基于游离二氧化硫 ==========
    if 'free_sulfur_dioxide' in df.columns:
        # 游离二氧化硫异常
        free_so2_75 = df['free_sulfur_dioxide'].quantile(0.75)
        high_free_so2_mask = df['free_sulfur_dioxide'] > free_so2_75
        missing_score[high_free_so2_mask] += 0.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_AvocadoRipeness(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于硬度（firmness） ==========
    if 'firmness' in df.columns:
        # 硬度极端值更容易缺失成熟度
        firmness_25 = df['firmness'].quantile(0.25)
        firmness_75 = df['firmness'].quantile(0.75)

        # 过软（过熟）更容易缺失
        too_soft_mask = df['firmness'] < firmness_25
        missing_score[too_soft_mask] += 2.5

        # 过硬（未熟）更容易缺失
        too_hard_mask = df['firmness'] > firmness_75
        missing_score[too_hard_mask] += 2.0

    # ========== 2. 基于颜色类别 ==========
    if 'color_category' in df.columns:
        # 自动识别罕见颜色类别
        color_counts = df['color_category'].value_counts()
        color_freq = color_counts / len(df)

        for color in color_counts.index:
            if color_freq[color] < 0.05:  # 罕见颜色
                mask = df['color_category'] == color
                missing_score[mask] += 2.5
            elif color_freq[color] < 0.15:  # 较少见颜色
                mask = df['color_category'] == color
                missing_score[mask] += 1.0

    # ========== 3. 基于声音分贝（sound_db） ==========
    if 'sound_db' in df.columns:
        # 声音异常（过高或过低）更容易缺失
        sound_25 = df['sound_db'].quantile(0.25)
        sound_75 = df['sound_db'].quantile(0.75)

        low_sound_mask = df['sound_db'] < sound_25
        missing_score[low_sound_mask] += 1.5

        high_sound_mask = df['sound_db'] > sound_75
        missing_score[high_sound_mask] += 1.5

        # 极高分贝（可能内部腐烂）
        sound_90 = df['sound_db'].quantile(0.90)
        extreme_sound_mask = df['sound_db'] > sound_90
        missing_score[extreme_sound_mask] += 1.0

    # ========== 4. 基于重量（weight_g） ==========
    if 'weight_g' in df.columns:
        # 重量极端值更容易缺失
        weight_25 = df['weight_g'].quantile(0.25)
        weight_75 = df['weight_g'].quantile(0.75)

        light_weight_mask = df['weight_g'] < weight_25
        missing_score[light_weight_mask] += 1.0

        heavy_weight_mask = df['weight_g'] > weight_75
        missing_score[heavy_weight_mask] += 1.0

    # ========== 5. 基于色相（hue） ==========
    if 'hue' in df.columns:
        # 色相极端值（绿色调或黄色调）更容易缺失
        hue_25 = df['hue'].quantile(0.25)
        hue_75 = df['hue'].quantile(0.75)

        low_hue_mask = df['hue'] < hue_25
        missing_score[low_hue_mask] += 1.5

        high_hue_mask = df['hue'] > hue_75
        missing_score[high_hue_mask] += 1.5

    # ========== 6. 基于饱和度（saturation） ==========
    if 'saturation' in df.columns:
        # 饱和度异常更容易缺失
        sat_25 = df['saturation'].quantile(0.25)
        sat_75 = df['saturation'].quantile(0.75)

        low_sat_mask = df['saturation'] < sat_25
        missing_score[low_sat_mask] += 1.0

        high_sat_mask = df['saturation'] > sat_75
        missing_score[high_sat_mask] += 1.0

    # ========== 7. 基于亮度（brightness） ==========
    if 'brightness' in df.columns:
        # 亮度异常（过暗或过亮）更容易缺失
        bright_25 = df['brightness'].quantile(0.25)
        bright_75 = df['brightness'].quantile(0.75)

        dark_mask = df['brightness'] < bright_25
        missing_score[dark_mask] += 1.0

        bright_mask = df['brightness'] > bright_75
        missing_score[bright_mask] += 1.0

    # ========== 8. 基于体积（size_cm3） ==========
    if 'size_cm3' in df.columns:
        # 体积异常更容易缺失
        size_25 = df['size_cm3'].quantile(0.25)
        size_75 = df['size_cm3'].quantile(0.75)

        small_size_mask = df['size_cm3'] < size_25
        missing_score[small_size_mask] += 1.0

        large_size_mask = df['size_cm3'] > size_75
        missing_score[large_size_mask] += 1.0

    # ========== 9. 基于多变量组合异常检测 ==========
    # 同时满足多个极端条件的样本更容易缺失
    extreme_count = np.zeros(n)

    # 检查是否同时处于多个极端状态
    if 'firmness' in df.columns:
        extreme_count += ((df['firmness'] < firmness_25) | (df['firmness'] > firmness_75)).astype(int)
    if 'hue' in df.columns:
        extreme_count += ((df['hue'] < hue_25) | (df['hue'] > hue_75)).astype(int)
    if 'sound_db' in df.columns:
        extreme_count += ((df['sound_db'] < sound_25) | (df['sound_db'] > sound_75)).astype(int)

    # 处于3个以上极端状态的样本更容易缺失
    multi_extreme_mask = extreme_count >= 3
    missing_score[multi_extreme_mask] += 2.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_concrete(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于水泥含量 ==========
    if 'cement' in df.columns:
        cement_25 = df['cement'].quantile(0.25)
        cement_75 = df['cement'].quantile(0.75)

        # 水泥含量过低（<25分位数）更容易缺失
        low_cement_mask = df['cement'] < cement_25
        missing_score[low_cement_mask] += 2.0

        # 水泥含量过高（>75分位数）更容易缺失
        high_cement_mask = df['cement'] > cement_75
        missing_score[high_cement_mask] += 1.5

    # ========== 2. 基于水含量 ==========
    if 'water' in df.columns:
        water_25 = df['water'].quantile(0.25)
        water_75 = df['water'].quantile(0.75)

        # 水含量过低更容易缺失
        low_water_mask = df['water'] < water_25
        missing_score[low_water_mask] += 2.0

        # 水含量过高更容易缺失
        high_water_mask = df['water'] > water_75
        missing_score[high_water_mask] += 2.0

    # ========== 3. 基于水灰比 ==========
    if 'cement' in df.columns and 'water' in df.columns:
        # 计算水灰比（水/水泥）
        water_cement_ratio = df['water'] / df['cement']

        wc_25 = water_cement_ratio.quantile(0.25)
        wc_75 = water_cement_ratio.quantile(0.75)

        # 水灰比过低（<0.3）更容易缺失
        low_wc_mask = water_cement_ratio < 0.3
        missing_score[low_wc_mask] += 2.5

        # 水灰比过高（>0.6）更容易缺失
        high_wc_mask = water_cement_ratio > 0.6
        missing_score[high_wc_mask] += 2.5

        # 正常水灰比（0.4-0.5）不容易缺失
        normal_wc_mask = (water_cement_ratio >= 0.4) & (water_cement_ratio <= 0.5)
        missing_score[normal_wc_mask] -= 0.5

    # ========== 4. 基于减水剂 ==========
    if 'superplasticizer' in df.columns:
        # 使用减水剂的配比更容易缺失
        has_superplasticizer = df['superplasticizer'] > 0
        missing_score[has_superplasticizer] += 1.5

        # 减水剂用量过高（>75分位数）更容易缺失
        sp_75 = df['superplasticizer'][df['superplasticizer'] > 0].quantile(0.75) if (
                    df['superplasticizer'] > 0).any() else 0
        high_sp_mask = df['superplasticizer'] > sp_75
        missing_score[high_sp_mask] += 1.0

    # ========== 5. 基于矿渣掺量 ==========
    if 'blast_furnace_slag' in df.columns:
        # 高矿渣掺量（>75分位数）更容易缺失
        slag_75 = df['blast_furnace_slag'].quantile(0.75)
        high_slag_mask = df['blast_furnace_slag'] > slag_75
        missing_score[high_slag_mask] += 1.5

        # 无矿渣的配比不容易缺失
        no_slag_mask = df['blast_furnace_slag'] == 0
        missing_score[no_slag_mask] -= 0.5

    # ========== 6. 基于粉煤灰掺量 ==========
    if 'fly_ash' in df.columns:
        # 高粉煤灰掺量（>75分位数）更容易缺失
        ash_75 = df['fly_ash'].quantile(0.75)
        high_ash_mask = df['fly_ash'] > ash_75
        missing_score[high_ash_mask] += 1.5

        # 无粉煤灰的配比不容易缺失
        no_ash_mask = df['fly_ash'] == 0
        missing_score[no_ash_mask] -= 0.5

    # ========== 7. 基于龄期 ==========
    if 'age' in df.columns:
        # 早期强度（<28天）更容易缺失
        early_age_mask = df['age'] < 28
        missing_score[early_age_mask] += 2.0

        # 极早期（<7天）更容易缺失
        very_early_mask = df['age'] < 7
        missing_score[very_early_mask] += 1.5

        # 长期强度（>90天）记录较完整
        long_age_mask = df['age'] > 90
        missing_score[long_age_mask] -= 0.5

    # ========== 8. 基于骨料含量 ==========
    if 'coarse_aggregate' in df.columns:
        agg_25 = df['coarse_aggregate'].quantile(0.25)
        agg_75 = df['coarse_aggregate'].quantile(0.75)

        low_agg_mask = df['coarse_aggregate'] < agg_25
        missing_score[low_agg_mask] += 1.0

        high_agg_mask = df['coarse_aggregate'] > agg_75
        missing_score[high_agg_mask] += 1.0

    if 'fine_aggregate' in df.columns:
        fine_25 = df['fine_aggregate'].quantile(0.25)
        fine_75 = df['fine_aggregate'].quantile(0.75)

        low_fine_mask = df['fine_aggregate'] < fine_25
        missing_score[low_fine_mask] += 0.5

        high_fine_mask = df['fine_aggregate'] > fine_75
        missing_score[high_fine_mask] += 0.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_CCPP(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于温度 (AT) - 使用分位数 ==========
    if 'AT' in df.columns:
        at_25 = df['AT'].quantile(0.25)
        at_75 = df['AT'].quantile(0.75)

        # 温度过低（<25分位数）更容易缺失
        low_temp_mask = df['AT'] < at_25
        missing_score[low_temp_mask] += 2.0

        # 温度过高（>75分位数）更容易缺失
        high_temp_mask = df['AT'] > at_75
        missing_score[high_temp_mask] += 2.0

        # 温度极低（<10分位数）更容易缺失
        at_10 = df['AT'].quantile(0.10)
        very_low_temp_mask = df['AT'] < at_10
        missing_score[very_low_temp_mask] += 0.5

        # 温度极高（>90分位数）更容易缺失
        at_90 = df['AT'].quantile(0.90)
        very_high_temp_mask = df['AT'] > at_90
        missing_score[very_high_temp_mask] += 0.5

    # ========== 2. 基于排气真空 (V) - 使用分位数 ==========
    if 'V' in df.columns:
        v_25 = df['V'].quantile(0.25)
        v_75 = df['V'].quantile(0.75)

        # 真空度过低（<25分位数）更容易缺失
        low_vacuum_mask = df['V'] < v_25
        missing_score[low_vacuum_mask] += 2.0

        # 真空度过高（>75分位数）更容易缺失
        high_vacuum_mask = df['V'] > v_75
        missing_score[high_vacuum_mask] += 2.0

        # 真空度极低（<10分位数）更容易缺失
        v_10 = df['V'].quantile(0.10)
        very_low_vacuum_mask = df['V'] < v_10
        missing_score[very_low_vacuum_mask] += 0.5

        # 真空度极高（>90分位数）更容易缺失
        v_90 = df['V'].quantile(0.90)
        very_high_vacuum_mask = df['V'] > v_90
        missing_score[very_high_vacuum_mask] += 0.5

    # ========== 3. 基于环境压力 (AP) - 使用分位数 ==========
    if 'AP' in df.columns:
        ap_25 = df['AP'].quantile(0.25)
        ap_75 = df['AP'].quantile(0.75)

        # 气压过低（<25分位数）更容易缺失
        low_pressure_mask = df['AP'] < ap_25
        missing_score[low_pressure_mask] += 1.5

        # 气压过高（>75分位数）更容易缺失
        high_pressure_mask = df['AP'] > ap_75
        missing_score[high_pressure_mask] += 1.5

        # 气压极低（<10分位数）更容易缺失
        ap_10 = df['AP'].quantile(0.10)
        very_low_pressure_mask = df['AP'] < ap_10
        missing_score[very_low_pressure_mask] += 1.0

        # 气压极高（>90分位数）更容易缺失
        ap_90 = df['AP'].quantile(0.90)
        very_high_pressure_mask = df['AP'] > ap_90
        missing_score[very_high_pressure_mask] += 1.0

    # ========== 4. 基于相对湿度 (RH) - 使用分位数 ==========
    if 'RH' in df.columns:
        rh_25 = df['RH'].quantile(0.25)
        rh_75 = df['RH'].quantile(0.75)

        # 湿度过低（<25分位数）更容易缺失
        low_humidity_mask = df['RH'] < rh_25
        missing_score[low_humidity_mask] += 1.0

        # 湿度过高（>75分位数）更容易缺失
        high_humidity_mask = df['RH'] > rh_75
        missing_score[high_humidity_mask] += 2.0

        # 湿度极高（>90分位数）更容易缺失
        rh_90 = df['RH'].quantile(0.90)
        very_high_humidity_mask = df['RH'] > rh_90
        missing_score[very_high_humidity_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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

    # 注入缺失值
    df.loc[missing_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_AirfoilSelfNoise(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于频率 (f) - 使用分位数 ==========
    if 'f' in df.columns:
        f_25 = df['f'].quantile(0.25)
        f_75 = df['f'].quantile(0.75)

        # 低频（<25分位数）更容易缺失
        low_freq_mask = df['f'] < f_25
        missing_score[low_freq_mask] += 1.5

        # 高频（>75分位数）更容易缺失
        high_freq_mask = df['f'] > f_75
        missing_score[high_freq_mask] += 2.0

        # 极高频（>90分位数）更容易缺失
        f_90 = df['f'].quantile(0.90)
        very_high_freq_mask = df['f'] > f_90
        missing_score[very_high_freq_mask] += 1.5

    # ========== 2. 基于攻角 (alpha) - 使用分位数 ==========
    if 'alpha' in df.columns:
        alpha_25 = df['alpha'].quantile(0.25)
        alpha_75 = df['alpha'].quantile(0.75)

        # 负攻角（<25分位数）更容易缺失
        negative_alpha_mask = df['alpha'] < alpha_25
        missing_score[negative_alpha_mask] += 1.5

        # 大攻角（>75分位数）更容易缺失
        high_alpha_mask = df['alpha'] > alpha_75
        missing_score[high_alpha_mask] += 2.5

        # 极大攻角（>90分位数）更容易缺失
        alpha_90 = df['alpha'].quantile(0.90)
        very_high_alpha_mask = df['alpha'] > alpha_90
        missing_score[very_high_alpha_mask] += 2.0

    # ========== 3. 基于弦长 (c) - 使用分位数 ==========
    if 'c' in df.columns:
        c_25 = df['c'].quantile(0.25)
        c_75 = df['c'].quantile(0.75)

        # 弦长过短（<25分位数）更容易缺失
        short_chord_mask = df['c'] < c_25
        missing_score[short_chord_mask] += 1.5

        # 弦长过长（>75分位数）更容易缺失
        long_chord_mask = df['c'] > c_75
        missing_score[long_chord_mask] += 1.5

        # 极长弦长（>90分位数）更容易缺失
        c_90 = df['c'].quantile(0.90)
        very_long_chord_mask = df['c'] > c_90
        missing_score[very_long_chord_mask] += 1.0

    # ========== 4. 基于来流速度 (U_infinity) - 使用分位数 ==========
    if 'U_infinity' in df.columns:
        u_25 = df['U_infinity'].quantile(0.25)
        u_75 = df['U_infinity'].quantile(0.75)

        # 低速（<25分位数）更容易缺失
        low_speed_mask = df['U_infinity'] < u_25
        missing_score[low_speed_mask] += 1.5

        # 高速（>75分位数）更容易缺失
        high_speed_mask = df['U_infinity'] > u_75
        missing_score[high_speed_mask] += 2.0

        # 极高速（>90分位数）更容易缺失
        u_90 = df['U_infinity'].quantile(0.90)
        very_high_speed_mask = df['U_infinity'] > u_90
        missing_score[very_high_speed_mask] += 1.5

    # ========== 5. 基于边界层厚度 (delta) - 使用分位数 ==========
    if 'delta' in df.columns:
        delta_25 = df['delta'].quantile(0.25)
        delta_75 = df['delta'].quantile(0.75)

        # 边界层过薄（<25分位数）更容易缺失
        thin_boundary_mask = df['delta'] < delta_25
        missing_score[thin_boundary_mask] += 1.0

        # 边界层过厚（>75分位数）更容易缺失
        thick_boundary_mask = df['delta'] > delta_75
        missing_score[thick_boundary_mask] += 1.5

        # 极厚边界层（>90分位数）更容易缺失
        delta_90 = df['delta'].quantile(0.90)
        very_thick_boundary_mask = df['delta'] > delta_90
        missing_score[very_thick_boundary_mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_Abalone(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于性别 ==========
    if 'Sex' in df.columns:
        # 幼体不容易缺失（通常年龄小，容易计数）
        infant_mask = df['Sex'] == 0
        missing_score[infant_mask] -= 0.5

    # ========== 2. 基于长度 (Length) ==========
    if 'Length' in df.columns:
        len_25 = df['Length'].quantile(0.25)
        len_75 = df['Length'].quantile(0.75)

        # 长度过小（<25分位数）更容易缺失
        short_mask = df['Length'] < len_25
        missing_score[short_mask] += 2.0

        # 长度过大（>75分位数）更容易缺失
        long_mask = df['Length'] > len_75
        missing_score[long_mask] += 1.5

    # ========== 3. 基于直径 (Diameter) ==========
    if 'Diameter' in df.columns:
        dia_25 = df['Diameter'].quantile(0.25)
        dia_75 = df['Diameter'].quantile(0.75)

        # 直径过小（<25分位数）更容易缺失
        small_dia_mask = df['Diameter'] < dia_25
        missing_score[small_dia_mask] += 2.0

        # 直径过大（>75分位数）更容易缺失
        large_dia_mask = df['Diameter'] > dia_75
        missing_score[large_dia_mask] += 1.5

    # ========== 4. 基于高度 (Height) ==========
    if 'Height' in df.columns:
        height_25 = df['Height'].quantile(0.25)
        height_75 = df['Height'].quantile(0.75)

        # 高度过矮（<25分位数）更容易缺失
        short_height_mask = df['Height'] < height_25
        missing_score[short_height_mask] += 2.0

        # 高度过高（>75分位数）更容易缺失
        tall_height_mask = df['Height'] > height_75
        missing_score[tall_height_mask] += 2.0

    # ========== 5. 基于整体重量 (Whole weight) ==========
    if 'Whole_weight' in df.columns:
        ww_25 = df['Whole_weight'].quantile(0.25)
        ww_75 = df['Whole_weight'].quantile(0.75)

        # 重量过轻（<25分位数）更容易缺失
        light_mask = df['Whole_weight'] < ww_25
        missing_score[light_mask] += 2.0

        # 重量过重（>75分位数）更容易缺失
        heavy_mask = df['Whole_weight'] > ww_75
        missing_score[heavy_mask] += 1.5

    # ========== 6. 基于肉重 (Shucked weight) ==========
    if 'Shucked_weight' in df.columns:
        sw_25 = df['Shucked_weight'].quantile(0.25)
        sw_75 = df['Shucked_weight'].quantile(0.75)

        # 肉重过轻（<25分位数）更容易缺失
        light_meat_mask = df['Shucked_weight'] < sw_25
        missing_score[light_meat_mask] += 2.0

        # 肉重过重（>75分位数）更容易缺失
        heavy_meat_mask = df['Shucked_weight'] > sw_75
        missing_score[heavy_meat_mask] += 1.5

    # ========== 7. 基于内脏重 (Viscera weight) ==========
    if 'Viscera_weight' in df.columns:
        vw_25 = df['Viscera_weight'].quantile(0.25)
        vw_75 = df['Viscera_weight'].quantile(0.75)

        # 内脏过轻（<25分位数）更容易缺失
        light_viscera_mask = df['Viscera_weight'] < vw_25
        missing_score[light_viscera_mask] += 1.5

        # 内脏过重（>75分位数）更容易缺失
        heavy_viscera_mask = df['Viscera_weight'] > vw_75
        missing_score[heavy_viscera_mask] += 1.5

    # ========== 8. 基于壳重 (Shell weight) - 最重要指标 ==========
    if 'Shell_weight' in df.columns:
        shell_25 = df['Shell_weight'].quantile(0.25)
        shell_75 = df['Shell_weight'].quantile(0.75)

        # 壳重过轻（<25分位数）更容易缺失
        light_shell_mask = df['Shell_weight'] < shell_25
        missing_score[light_shell_mask] += 2.5

        # 壳重过重（>75分位数）更容易缺失
        heavy_shell_mask = df['Shell_weight'] > shell_75
        missing_score[heavy_shell_mask] += 2.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_ParisHousing(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于面积 (squareMeters) ==========
    if 'squareMeters' in df.columns:
        sq_25 = df['squareMeters'].quantile(0.25)
        sq_75 = df['squareMeters'].quantile(0.75)

        # 面积过小（<25分位数）更容易缺失
        small_area_mask = df['squareMeters'] < sq_25
        missing_score[small_area_mask] += 2.0

        # 面积过大（>75分位数）更容易缺失
        large_area_mask = df['squareMeters'] > sq_75
        missing_score[large_area_mask] += 2.0

        # 极大面积（>90分位数）更容易缺失
        sq_90 = df['squareMeters'].quantile(0.90)
        very_large_area_mask = df['squareMeters'] > sq_90
        missing_score[very_large_area_mask] += 1.5

    # ========== 2. 基于房间数 (numberOfRooms) ==========
    if 'numberOfRooms' in df.columns:
        rooms_25 = df['numberOfRooms'].quantile(0.25)
        rooms_75 = df['numberOfRooms'].quantile(0.75)

        # 房间数过少（<25分位数）更容易缺失
        few_rooms_mask = df['numberOfRooms'] < rooms_25
        missing_score[few_rooms_mask] += 1.5

        # 房间数过多（>75分位数）更容易缺失
        many_rooms_mask = df['numberOfRooms'] > rooms_75
        missing_score[many_rooms_mask] += 2.0

        # 极多房间（>90分位数）更容易缺失
        rooms_90 = df['numberOfRooms'].quantile(0.90)
        very_many_rooms_mask = df['numberOfRooms'] > rooms_90
        missing_score[very_many_rooms_mask] += 1.5

    # ========== 3. 基于楼层 (floors) ==========
    if 'floors' in df.columns:
        floors_25 = df['floors'].quantile(0.25)
        floors_75 = df['floors'].quantile(0.75)

        # 低楼层（<25分位数）更容易缺失
        low_floor_mask = df['floors'] < floors_25
        missing_score[low_floor_mask] += 1.0

        # 高楼层（>75分位数）更容易缺失
        high_floor_mask = df['floors'] > floors_75
        missing_score[high_floor_mask] += 1.5

        # 极高楼层（>90分位数）更容易缺失
        floors_90 = df['floors'].quantile(0.90)
        very_high_floor_mask = df['floors'] > floors_90
        missing_score[very_high_floor_mask] += 1.0

    # ========== 4. 基于城市代码 (cityCode) ==========
    if 'cityCode' in df.columns:
        # 计算每个城市的出现频率
        city_counts = df['cityCode'].value_counts()
        city_freq = city_counts / len(df)

        # 罕见城市（出现频率低于5%）更容易缺失
        for city in city_counts.index:
            if city_freq[city] < 0.05:
                mask = df['cityCode'] == city
                missing_score[mask] += 2.5
            elif city_freq[city] < 0.10:
                mask = df['cityCode'] == city
                missing_score[mask] += 1.5

    # ========== 5. 基于建造年份 (made) ==========
    if 'made' in df.columns:
        made_25 = df['made'].quantile(0.25)
        made_75 = df['made'].quantile(0.75)

        # 老旧房屋（<25分位数）更容易缺失
        old_house_mask = df['made'] < made_25
        missing_score[old_house_mask] += 2.0

        # 新房（>75分位数）更容易缺失
        new_house_mask = df['made'] > made_75
        missing_score[new_house_mask] += 1.5

    # ========== 6. 基于是否有泳池 (hasPool) ==========
    if 'hasPool' in df.columns:
        # 有泳池的房屋不容易缺失
        has_pool_mask = df['hasPool'] == 1
        missing_score[has_pool_mask] -= 0.5

        # 无泳池的房屋更容易缺失
        no_pool_mask = df['hasPool'] == 0
        missing_score[no_pool_mask] += 0.5

    # ========== 7. 基于是否有花园 (hasYard) ==========
    if 'hasYard' in df.columns:
        # 有花园的房屋不容易缺失
        has_yard_mask = df['hasYard'] == 1
        missing_score[has_yard_mask] -= 0.5

        # 无花园的房屋更容易缺失
        no_yard_mask = df['hasYard'] == 0
        missing_score[no_yard_mask] += 1.0

    # ========== 8. 是否新建筑 (isNewBuilt) ==========
    if 'isNewBuilt' in df.columns:
        # 新建筑不容易缺失
        new_built_mask = df['isNewBuilt'] == 1
        missing_score[new_built_mask] -= 0.5

        # 非新建筑更容易缺失
        not_new_built_mask = df['isNewBuilt'] == 0
        missing_score[not_new_built_mask] += 0.5

    # ========== 9. 基于地下室面积 (basement) ==========
    if 'basement' in df.columns:
        bsmt_25 = df['basement'].quantile(0.25)
        bsmt_75 = df['basement'].quantile(0.75)

        # 无地下室或面积过小更容易缺失
        no_basement_mask = df['basement'] == 0
        missing_score[no_basement_mask] += 1.0

        # 大面积地下室更容易缺失
        large_basement_mask = df['basement'] > bsmt_75
        missing_score[large_basement_mask] += 1.5

    # ========== 10. 基于车库面积 (garage) ==========
    if 'garage' in df.columns:
        garage_25 = df['garage'].quantile(0.25)
        garage_75 = df['garage'].quantile(0.75)

        # 无车库更容易缺失
        no_garage_mask = df['garage'] == 0
        missing_score[no_garage_mask] += 1.0

        # 大面积车库更容易缺失
        large_garage_mask = df['garage'] > garage_75
        missing_score[large_garage_mask] += 1.0

    # ========== 11. 基于前业主数 (numPrevOwners) ==========
    if 'numPrevOwners' in df.columns:
        owners_75 = df['numPrevOwners'].quantile(0.75)
        owners_90 = df['numPrevOwners'].quantile(0.90)

        # 多次交易的房屋更容易缺失
        many_owners_mask = df['numPrevOwners'] > owners_75
        missing_score[many_owners_mask] += 1.5

        # 极多次交易的房屋更容易缺失
        very_many_owners_mask = df['numPrevOwners'] > owners_90
        missing_score[very_many_owners_mask] += 1.0


    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_ETT(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于高压负荷 (HUFL) ==========
    if 'HUFL' in df.columns:
        hufl_25 = df['HUFL'].quantile(0.25)
        hufl_75 = df['HUFL'].quantile(0.75)

        # 负荷过低（<25分位数）更容易缺失
        low_hufl_mask = df['HUFL'] < hufl_25
        missing_score[low_hufl_mask] += 2.0

        # 负荷过高（>75分位数）更容易缺失
        high_hufl_mask = df['HUFL'] > hufl_75
        missing_score[high_hufl_mask] += 2.0

        # 极高负荷（>90分位数）更容易缺失
        hufl_90 = df['HUFL'].quantile(0.90)
        very_high_hufl_mask = df['HUFL'] > hufl_90
        missing_score[very_high_hufl_mask] += 1.5

    # ========== 2. 基于高压负荷 (HULL) ==========
    if 'HULL' in df.columns:
        hull_25 = df['HULL'].quantile(0.25)
        hull_75 = df['HULL'].quantile(0.75)

        low_hull_mask = df['HULL'] < hull_25
        missing_score[low_hull_mask] += 1.0

        high_hull_mask = df['HULL'] > hull_75
        missing_score[high_hull_mask] += 2.0

    # ========== 3. 基于中压负荷 (MUFL) ==========
    if 'MUFL' in df.columns:
        mufl_25 = df['MUFL'].quantile(0.25)
        mufl_75 = df['MUFL'].quantile(0.75)

        low_mufl_mask = df['MUFL'] < mufl_25
        missing_score[low_mufl_mask] += 1.0

        high_mufl_mask = df['MUFL'] > mufl_75
        missing_score[high_mufl_mask] += 1.5

    # ========== 4. 基于中压负荷 (MULL) ==========
    if 'MULL' in df.columns:
        mull_75 = df['MULL'].quantile(0.75)
        high_mull_mask = df['MULL'] > mull_75
        missing_score[high_mull_mask] += 1.5

    # ========== 5. 基于低压负荷 (LUFL) ==========
    if 'LUFL' in df.columns:
        lufl_25 = df['LUFL'].quantile(0.25)
        lufl_75 = df['LUFL'].quantile(0.75)

        low_lufl_mask = df['LUFL'] < lufl_25
        missing_score[low_lufl_mask] += 1.0

        high_lufl_mask = df['LUFL'] > lufl_75
        missing_score[high_lufl_mask] += 1.5

    # ========== 6. 基于低压负荷 (LULL) ==========
    if 'LULL' in df.columns:
        lull_75 = df['LULL'].quantile(0.75)
        high_lull_mask = df['LULL'] > lull_75
        missing_score[high_lull_mask] += 1.5


    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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
    df = df.drop(columns=['hour', 'is_night', 'is_peak'], errors='ignore')
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Illness(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于儿童病例 (AGE_0_TO_4) ==========
    if 'AGE_0_TO_4' in df.columns:
        child_25 = df['AGE_0_TO_4'].quantile(0.25)
        child_75 = df['AGE_0_TO_4'].quantile(0.75)

        # 儿童病例过低（<25分位数）更容易缺失
        low_child_mask = df['AGE_0_TO_4'] < child_25
        missing_score[low_child_mask] += 1.5

        # 儿童病例过高（>75分位数）更容易缺失
        high_child_mask = df['AGE_0_TO_4'] > child_75
        missing_score[high_child_mask] += 2.0

        # 极高儿童病例（>90分位数）更容易缺失
        child_90 = df['AGE_0_TO_4'].quantile(0.90)
        very_high_child_mask = df['AGE_0_TO_4'] > child_90
        missing_score[very_high_child_mask] += 1.5

    # ========== 2. 基于青少年病例 (AGE_5_TO_24) ==========
    if 'AGE_5_TO_24' in df.columns:
        youth_25 = df['AGE_5_TO_24'].quantile(0.25)
        youth_75 = df['AGE_5_TO_24'].quantile(0.75)

        low_youth_mask = df['AGE_5_TO_24'] < youth_25
        missing_score[low_youth_mask] += 1.0

        high_youth_mask = df['AGE_5_TO_24'] > youth_75
        missing_score[high_youth_mask] += 2.0

        # 极高青少年病例（>90分位数）
        youth_90 = df['AGE_5_TO_24'].quantile(0.90)
        very_high_youth_mask = df['AGE_5_TO_24'] > youth_90
        missing_score[very_high_youth_mask] += 1.5

    # ========== 3. 基于加权ILI (WEIGHTED_ILI) ==========
    if 'WEIGHTED_ILI' in df.columns:
        ili_w_25 = df['WEIGHTED_ILI'].quantile(0.25)
        ili_w_75 = df['WEIGHTED_ILI'].quantile(0.75)

        # ILI过低（<25分位数）更容易缺失
        low_ili_mask = df['WEIGHTED_ILI'] < ili_w_25
        missing_score[low_ili_mask] += 1.5

        # ILI过高（>75分位数）更容易缺失
        high_ili_mask = df['WEIGHTED_ILI'] > ili_w_75
        missing_score[high_ili_mask] += 2.5

        # 极高ILI（>90分位数）
        ili_w_90 = df['WEIGHTED_ILI'].quantile(0.90)
        very_high_ili_mask = df['WEIGHTED_ILI'] > ili_w_90
        missing_score[very_high_ili_mask] += 2.0

    # ========== 4. 基于未加权ILI (UNWEIGHTED_ILI) ==========
    if 'UNWEIGHTED_ILI' in df.columns:
        ili_u_25 = df['UNWEIGHTED_ILI'].quantile(0.25)
        ili_u_75 = df['UNWEIGHTED_ILI'].quantile(0.75)

        low_ili_u_mask = df['UNWEIGHTED_ILI'] < ili_u_25
        missing_score[low_ili_u_mask] += 1.0

        high_ili_u_mask = df['UNWEIGHTED_ILI'] > ili_u_75
        missing_score[high_ili_u_mask] += 2.0

    # ========== 5. 基于总病例数 (ILITOTAL) ==========
    if 'ILITOTAL' in df.columns:
        total_25 = df['ILITOTAL'].quantile(0.25)
        total_75 = df['ILITOTAL'].quantile(0.75)

        low_total_mask = df['ILITOTAL'] < total_25
        missing_score[low_total_mask] += 1.0

        high_total_mask = df['ILITOTAL'] > total_75
        missing_score[high_total_mask] += 2.0

        # 极高总病例（>90分位数）
        total_90 = df['ILITOTAL'].quantile(0.90)
        very_high_total_mask = df['ILITOTAL'] > total_90
        missing_score[very_high_total_mask] += 1.5

    # ========== 6. 基于医疗机构数量 (NUM_OF_PROVIDERS) ==========
    if 'NUM_OF_PROVIDERS' in df.columns:
        providers_25 = df['NUM_OF_PROVIDERS'].quantile(0.25)
        providers_75 = df['NUM_OF_PROVIDERS'].quantile(0.75)

        # 医疗机构数量少（<25分位数）更容易缺失
        low_providers_mask = df['NUM_OF_PROVIDERS'] < providers_25
        missing_score[low_providers_mask] += 2.0

        # 医疗机构数量多（>75分位数）不容易缺失
        high_providers_mask = df['NUM_OF_PROVIDERS'] > providers_75
        missing_score[high_providers_mask] -= 0.5

        # 医疗机构数量极少（<10分位数）
        providers_10 = df['NUM_OF_PROVIDERS'].quantile(0.10)
        very_low_providers_mask = df['NUM_OF_PROVIDERS'] < providers_10
        missing_score[very_low_providers_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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
    df = df.drop(columns=['month', 'is_flu_season', 'is_non_flu_season'], errors='ignore')
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


def insert_null_Exchange(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # 特征列列表
    feature_cols = ['Num0', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']

    # ========== 1. 基于各维度特征 ==========
    for col in feature_cols:
        if col in df.columns:
            col_25 = df[col].quantile(0.25)
            col_75 = df[col].quantile(0.75)

            # 特征值过低（<25分位数）更容易缺失
            low_mask = df[col] < col_25
            missing_score[low_mask] += 1.0

            # 特征值过高（>75分位数）更容易缺失
            high_mask = df[col] > col_75
            missing_score[high_mask] += 1.5

            # 极高值（>90分位数）更容易缺失
            col_90 = df[col].quantile(0.90)
            very_high_mask = df[col] > col_90
            missing_score[very_high_mask] += 1.0

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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


def insert_null_Weather(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    n = len(df)
    n_missing = int(len(df) * rate / 100)
    missing_score = np.zeros(n)

    # ========== 1. 基于温度 (T) ==========
    if 'T' in df.columns:
        t_25 = df['T'].quantile(0.25)
        t_75 = df['T'].quantile(0.75)

        # 温度过低（<25分位数）更容易缺失
        low_temp_mask = df['T'] < t_25
        missing_score[low_temp_mask] += 1.5

        # 温度过高（>75分位数）更容易缺失
        high_temp_mask = df['T'] > t_75
        missing_score[high_temp_mask] += 2.0

        # 极低温度（<10分位数）
        t_10 = df['T'].quantile(0.10)
        very_low_temp_mask = df['T'] < t_10
        missing_score[very_low_temp_mask] += 1.5

        # 极高温度（>90分位数）
        t_90 = df['T'].quantile(0.90)
        very_high_temp_mask = df['T'] > t_90
        missing_score[very_high_temp_mask] += 1.5

    # ========== 2. 基于相对湿度 (rh) ==========
    if 'rh' in df.columns:
        rh_25 = df['rh'].quantile(0.25)
        rh_75 = df['rh'].quantile(0.75)

        # 湿度过低（<25分位数）容易缺失
        low_rh_mask = df['rh'] < rh_25
        missing_score[low_rh_mask] += 1.0

        # 湿度过高（>75分位数）更容易缺失
        high_rh_mask = df['rh'] > rh_75
        missing_score[high_rh_mask] += 2.0

        # 极高湿度（>90分位数）
        rh_90 = df['rh'].quantile(0.90)
        very_high_rh_mask = df['rh'] > rh_90
        missing_score[very_high_rh_mask] += 1.5

    # ========== 3. 基于气压 (p) ==========
    if 'p' in df.columns:
        p_25 = df['p'].quantile(0.25)
        p_75 = df['p'].quantile(0.75)

        # 气压过低（<25分位数）更容易缺失
        low_p_mask = df['p'] < p_25
        missing_score[low_p_mask] += 1.5

        # 气压过高（>75分位数）容易缺失
        high_p_mask = df['p'] > p_75
        missing_score[high_p_mask] += 1.0

        # 气压剧烈变化
        p_diff = df['p'].diff().abs()
        p_diff_75 = p_diff.quantile(0.75)
        high_p_change_mask = p_diff > p_diff_75
        missing_score[high_p_change_mask] += 1.5

    # ========== 4. 基于风速 (wv, maxwv) ==========
    if 'wv' in df.columns:
        wv_75 = df['wv'].quantile(0.75)
        wv_90 = df['wv'].quantile(0.90)

        # 风速过高（>75分位数）更容易缺失
        high_wind_mask = df['wv'] > wv_75
        missing_score[high_wind_mask] += 2.0

        # 极高风速（>90分位数）
        very_high_wind_mask = df['wv'] > wv_90
        missing_score[very_high_wind_mask] += 1.5

    if 'maxwv' in df.columns:
        maxwv_75 = df['maxwv'].quantile(0.75)
        high_maxwind_mask = df['maxwv'] > maxwv_75
        missing_score[high_maxwind_mask] += 1.5

    # ========== 6. 基于太阳辐射 (SWDR, PAR) ==========
    if 'SWDR' in df.columns:
        swdr_75 = df['SWDR'].quantile(0.75)
        swdr_90 = df['SWDR'].quantile(0.90)

        # 高辐射时更容易缺失
        high_swdr_mask = df['SWDR'] > swdr_75
        missing_score[high_swdr_mask] += 1.5

        # 极高辐射时更容易缺失
        very_high_swdr_mask = df['SWDR'] > swdr_90
        missing_score[very_high_swdr_mask] += 1.0

    if 'PAR' in df.columns:
        par_75 = df['PAR'].quantile(0.75)
        high_par_mask = df['PAR'] > par_75
        missing_score[high_par_mask] += 1.5

    # ========== 处理缺失值导致的NaN ==========
    missing_score = np.nan_to_num(missing_score, nan=0.0)

    # ========== 归一化缺失概率分数 ==========
    if missing_score.min() < 0:
        missing_score = missing_score - missing_score.min()

    if missing_score.max() > 0:
        missing_score = missing_score / missing_score.max()
    else:
        missing_score = np.ones(n) / n

    # 添加随机噪声
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
    df = df.drop(columns=['hour', 'is_night', 'is_day'], errors='ignore')
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
    output_path = os.path.join(base_path, dataset, "null", "MAR")
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

        elif dataset == "concrete":
            insert_null_concrete(input_file, output_file, rate, target_column)
        elif dataset == "CCPP":
            insert_null_CCPP(input_file, output_file, rate, target_column)
        elif dataset == "AirfoilSelfNoise":
            insert_null_AirfoilSelfNoise(input_file, output_file, rate, target_column)
        elif dataset == "Abalone":
            insert_null_Abalone(input_file, output_file, rate, target_column)
        elif dataset == "ParisHousing":
            insert_null_ParisHousing(input_file, output_file, rate, target_column)

        elif dataset == "ETTh1" or dataset == "ETTm1":
            insert_null_ETT(input_file, output_file, rate, target_column)
        elif dataset == "Illness":
            insert_null_Illness(input_file, output_file, rate, target_column)
        elif dataset == "Exchange":
            insert_null_Exchange(input_file, output_file, rate, target_column)
        elif dataset == "Weather":
            insert_null_Weather(input_file, output_file, rate, target_column)
