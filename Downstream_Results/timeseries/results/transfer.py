import pandas as pd
import re


def convert_txt_to_csv_v2(input_txt_path, output_csv_path):
    """
    将原始txt数据文件转换为指定格式的CSV文件（改进版）
    确保输出为5列，使用逗号分隔

    参数:
        input_txt_path: 输入的txt文件路径
        output_csv_path: 输出的csv文件路径
    """

    data_rows = []

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 处理clean.csv的特殊格式
            if 'clean.csv' in line and '-' in line and 'RMSE:' in line:
                match = re.match(r'clean\.csv\s*-\s*RMSE:\s*([\d.]+),\s*MAE:\s*([\d.]+)', line)
                if match:
                    filename = 'clean.csv'
                    rmse = f"{float(match.group(1))}".rstrip('0').rstrip('.')
                    mae = f"{float(match.group(2))}".rstrip('0').rstrip('.')
                    pg_rmse = '0'
                    pg_mae = '0'
                    data_rows.append([filename, rmse, mae, pg_rmse, pg_mae])
                continue

            # 处理其他行（空格分隔）
            elif 'dirty' in line:
                parts = line.split()
                if len(parts) >= 3:
                    filename = parts[0]
                    rmse_val = float(parts[1])
                    mae_val = float(parts[2])

                    # 格式化RMSE和MAE，去掉末尾多余的0
                    rmse = f"{rmse_val}".rstrip('0').rstrip('.')
                    mae = f"{mae_val}".rstrip('0').rstrip('.')

                    # 检查是否有PG值
                    if len(parts) >= 5:
                        pg_rmse_val = float(parts[3])
                        pg_mae_val = float(parts[4])
                        # 格式化PG值，如果是0则显示为0
                        if pg_rmse_val == 0:
                            pg_rmse = '0'
                        else:
                            pg_rmse = f"{pg_rmse_val}".rstrip('0').rstrip('.')

                        if pg_mae_val == 0:
                            pg_mae = '0'
                        else:
                            pg_mae = f"{pg_mae_val}".rstrip('0').rstrip('.')
                    else:
                        pg_rmse = ''
                        pg_mae = ''

                    data_rows.append([filename, rmse, mae, pg_rmse, pg_mae])

    # 创建DataFrame（所有数据已经是字符串格式）
    df = pd.DataFrame(data_rows, columns=['File Name', 'RMSE', 'MAE', 'PG(RMSE)', 'PG(MAE)'])

    # 保存为CSV文件，使用逗号分隔（标准CSV格式）
    df.to_csv(output_csv_path, sep=',', index=False, encoding='utf-8')

    # 打印前几行确认格式
    print("\n输出文件前5行预览：")
    print(df.head().to_string())
    print(f"\nDataFrame形状: {df.shape} (行, 列)")
    print(f"列数: {len(df.columns)}")

    print(f"\n转换完成！")
    print(f"输入文件: {input_txt_path}")
    print(f"输出文件: {output_csv_path}")
    print(f"共处理 {len(df)} 行数据")

    return df


# 如果需要制表符分隔的版本
def convert_txt_to_csv_v2_tab(input_txt_path, output_csv_path):
    """
    使用制表符分隔的版本
    """
    data_rows = []

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if 'clean.csv' in line and '-' in line and 'RMSE:' in line:
                match = re.match(r'clean\.csv\s*-\s*RMSE:\s*([\d.]+),\s*MAE:\s*([\d.]+)', line)
                if match:
                    filename = 'clean.csv'
                    rmse = f"{float(match.group(1))}".rstrip('0').rstrip('.')
                    mae = f"{float(match.group(2))}".rstrip('0').rstrip('.')
                    pg_rmse = '0'
                    pg_mae = '0'
                    data_rows.append([filename, rmse, mae, pg_rmse, pg_mae])
                continue

            parts = line.split()
            if len(parts) >= 3:
                filename = parts[0]
                rmse_val = float(parts[1])
                mae_val = float(parts[2])

                rmse = f"{rmse_val}".rstrip('0').rstrip('.')
                mae = f"{mae_val}".rstrip('0').rstrip('.')

                if len(parts) >= 5:
                    pg_rmse_val = float(parts[3])
                    pg_mae_val = float(parts[4])
                    pg_rmse = '0' if pg_rmse_val == 0 else f"{pg_rmse_val}".rstrip('0').rstrip('.')
                    pg_mae = '0' if pg_mae_val == 0 else f"{pg_mae_val}".rstrip('0').rstrip('.')
                else:
                    pg_rmse = ''
                    pg_mae = ''

                data_rows.append([filename, rmse, mae, pg_rmse, pg_mae])

    df = pd.DataFrame(data_rows, columns=['File Name', 'RMSE', 'MAE', 'PG(RMSE)', 'PG(MAE)'])

    # 使用制表符分隔
    df.to_csv(output_csv_path, sep='\t', index=False, encoding='utf-8')

    print(f"\n转换完成（制表符分隔）！")
    print(f"输出文件: {output_csv_path}")
    print(f"共处理 {len(df)} 行，{len(df.columns)} 列")

    return df


# 使用示例
if __name__ == "__main__":
    input_file = 'micn-ettm1-mnar.txt'
    output_file = 'micn-ettm1-mnar.csv'

    # 使用逗号分隔（标准CSV格式）
    convert_txt_to_csv_v2(input_file, output_file)

    # 或者使用制表符分隔
    # convert_txt_to_csv_v2_tab(input_file, 'output_tab.csv')