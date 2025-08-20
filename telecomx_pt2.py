import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Carregue o DataFrame 'dados_tratados.csv'
# Suponha que o arquivo já foi salvo e contém as colunas limpas.
df = pd.read_csv('dados_tratados.csv')

# Identifique as colunas categóricas a serem codificadas
colunas_categoricas = [
    'Churn', 'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Inicialize o OneHotEncoder
# 'sparse_output=False' garante que a saída será um array denso (não esparso)
ohe = OneHotEncoder(sparse_output=False)

# Aplique a codificação one-hot nas colunas categóricas
df_encoded_cols = ohe.fit_transform(df[colunas_categoricas])

# Crie um DataFrame com as novas colunas codificadas
# 'ohe.get_feature_names_out()' cria os nomes das colunas de forma automática
df_encoded = pd.DataFrame(df_encoded_cols, columns=ohe.get_feature_names_out(colunas_categoricas))

# Junte o novo DataFrame codificado com as colunas numéricas originais
# 'axis=1' garante que a concatenação seja feita lado a lado
df_final = pd.concat([df.drop(columns=colunas_categoricas), df_encoded], axis=1)

# Verifique o resultado final do DataFrame
print(df_final.info())
print("\nPrimeiras 5 linhas do DataFrame codificado:")
print(df_final.head())