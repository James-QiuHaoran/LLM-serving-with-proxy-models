import pandas as pd


# read CSV file to dataframe
df = pd.read_csv('AzureLLMInferenceTrace_conv.csv')
print(df.head())

# the TIMESTAMP column contains the timestamps in the format of "2023-11-16 18:17:03.9799600"
# we need to convert it to the format of integer (number of microseconds since the epoch)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f').astype(int) // 10**6
print(df.head())

# write the converted dataframe to a new CSV file
df.to_csv('AzureLLMInferenceTrace_conv_int.csv', index=False)


# df = pd.read_csv('AzureLLMInferenceTrace_code.csv')
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f').astype(int) // 10**6
# df.to_csv('AzureLLMInferenceTrace_code_int.csv', index=False)