import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy.copy()
    target_observed_count = df[target_column].notnull().sum()
    missing_indices = df[df[target_column].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_target_codes = df_drop[target_column].values

    if unrelated_column != "None":
        df_filled = df.drop(columns=[unrelated_column])
    else:
        df_filled = df
    columns_to_fill = df_filled.columns.difference([target_column])
    df_filled[columns_to_fill] = df_filled[columns_to_fill].fillna(0)

    def impute_iim(df_filled):
        df_imputed = df_filled
        feature = target_column
        for i in range(len(df_filled)):
            if pd.isnull(df.iloc[i][feature]):
                print("---------------------")
                models = []
                errors = []
                for k in range(5, target_observed_count + 1,10):
                    knn = KNeighborsRegressor(n_neighbors=k)
                    X_train = df_filled.dropna(subset=[feature]).drop(columns=[feature])
                    y_train = df_filled.dropna(subset=[feature])[feature]
                    knn.fit(X_train.values, y_train.values)
                    distances, indices = knn.kneighbors(df_filled.iloc[i].drop(feature).values.reshape(1, -1))

                    X_train_neighbor = X_train.iloc[indices[0]]
                    y_train_neighbor = y_train.iloc[indices[0]]
                    model = LinearRegression()
                    model.fit(X_train_neighbor, y_train_neighbor)
                    models.append(model)

                    y_pred_neighbor = model.predict(X_train_neighbor)
                    error = mean_squared_error(y_train_neighbor, y_pred_neighbor)
                    errors.append(error)

                best_model_idx = errors.index(min(errors))
                best_model = models[best_model_idx]

                df_imputed.at[i,feature] = best_model.predict(df_filled.iloc[i].drop(feature).to_frame().T)[0]
                print(df_imputed.at[i,feature])

        return df_imputed

    df_imputed = impute_iim(df_filled)
    for index in missing_indices:
        generated_value = df_imputed.at[index, target_column]
        min_code = np.min(known_target_codes)
        max_code = np.max(known_target_codes)

        if generated_value < min_code:
            generated_value = int(min_code)
        elif generated_value > max_code:
            generated_value = int(max_code)
        #else:
        #    distances = np.abs(known_city_codes - generated_value)
        #    closest_index = np.argmin(distances)
        #    generated_value = int(known_city_codes[closest_index])
        target_name = label_encoders[target_column].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, target_column] = target_name
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)