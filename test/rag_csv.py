import pandas as pd

from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def init_model():
    gpt_chat_version = 'gpt-4o'
    gpt_config = get_model_configuration(gpt_chat_version)

    chat_model = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    return chat_model

if __name__ == "__main__":
    chat_model= init_model()

    df = pd.read_csv("example_csv.csv", skiprows=1)
    # print(df.head())
    agent = create_pandas_dataframe_agent(llm=chat_model,
                                      df=df,
                                      prefix='回答請使用繁體中文',
                                      agent_type="openai-tools",
                                      allow_dangerous_code=True, # langchain 2.0 要求加上這一行
                                      verbose=True)
    result = agent.invoke({"3月收盤價的平均值是?"})
    print(result['output'])
    print(df['收盤價'].mean())