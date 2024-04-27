import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Função para converter opções "Sim" e "Não" em 0 e 1
def sim_nao_para_binario(valor):
    if valor == "Sim":
        return 1
    else:
        return 0

# Função para previsão com base no Threshold
def prever_com_threshold(modelo, entrada, threshold):
    probabilidade = modelo.predict_proba(entrada)[0][1]
    if probabilidade >= threshold:
        return 1
    else:
        return 0

# Carregar o modelo treinado
model = joblib.load('pickle_rf_pycaret.pkl')

# Título do Dashboard
st.title('Previsão de Resposta do Cliente')

# Layout da interface
st.sidebar.title('Menu')
tabs = st.sidebar.radio("Escolha uma aba:", ('Online', 'Analytics'))

# Seção Online
if tabs == 'Online':
    st.subheader('Entradas do Usuário')
    st.write('Insira os dados do cliente:')
    accepted_cmp1 = st.selectbox('AcceptedCmp1', ['Sim', 'Não'])
    accepted_cmp2 = st.selectbox('AcceptedCmp2', ['Sim', 'Não'])
    accepted_cmp3 = st.selectbox('AcceptedCmp3', ['Sim', 'Não'])
    accepted_cmp4 = st.selectbox('AcceptedCmp4', ['Sim', 'Não'])
    accepted_cmp5 = st.selectbox('AcceptedCmp5', ['Sim', 'Não'])
    age = st.number_input('Age', value=0, step=1)
    complain = st.selectbox('Complain', ['Sim', 'Não'])
    education = st.selectbox('Education', ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'])
    income = st.number_input('Income', value=0, step=1)
    kidhome = st.number_input('Kidhome', value=0, step=1)
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Together', 'Widow', 'Absurd'])
    mnt_fish_products = st.number_input('MntFishProducts', value=0, step=1)
    mnt_fruits = st.number_input('MntFruits', value=0, step=1)
    mnt_gold_prods = st.number_input('MntGoldProds', value=0, step=1)
    mnt_meat_products = st.number_input('MntMeatProducts', value=0, step=1)
    mnt_sweet_products = st.number_input('MntSweetProducts', value=0, step=1)
    mnt_wines = st.number_input('MntWines', value=0, step=1)
    num_catalog_purchases = st.number_input('NumCatalogPurchases', value=0, step=1)
    num_deals_purchases = st.number_input('NumDealsPurchases', value=0, step=1)
    num_store_purchases = st.number_input('NumStorePurchases', value=0, step=1)
    num_web_purchases = st.number_input('NumWebPurchases', value=0, step=1)
    num_web_visits_month = st.number_input('NumWebVisitsMonth', value=0, step=1)
    recency = st.number_input('Recency', value=0, step=1)
    teenhome = st.number_input('Teenhome', value=0, step=1)
    time_customer = st.date_input('Time_Customer')

    threshold_online = st.slider("Escolha do Threshold:", 0.0, 1.0, 0.5, step=0.05)

    # Verificar se todos os campos foram preenchidos
    if st.button('Prever'):
        # Processamento dos dados de entrada
        data = {
            'AcceptedCmp1': sim_nao_para_binario(accepted_cmp1),
            'AcceptedCmp2': sim_nao_para_binario(accepted_cmp2),
            'AcceptedCmp3': sim_nao_para_binario(accepted_cmp3),
            'AcceptedCmp4': sim_nao_para_binario(accepted_cmp4),
            'AcceptedCmp5': sim_nao_para_binario(accepted_cmp5),
            'Age': age,
            'Complain': sim_nao_para_binario(complain),
            'Education': education,
            'Income': income,
            'Kidhome': kidhome,
            'Marital_Status': marital_status,
            'MntFishProducts': mnt_fish_products,
            'MntFruits': mnt_fruits,
            'MntGoldProds': mnt_gold_prods,
            'MntMeatProducts': mnt_meat_products,
            'MntSweetProducts': mnt_sweet_products,
            'MntWines': mnt_wines,
            'NumCatalogPurchases': num_catalog_purchases,
            'NumDealsPurchases': num_deals_purchases,
            'NumStorePurchases': num_store_purchases,
            'NumWebPurchases': num_web_purchases,
            'NumWebVisitsMonth': num_web_visits_month,
            'Recency': recency,
            'Teenhome': teenhome,
            'Time_Customer': (datetime.now().date() - time_customer).days  # Calcula a diferença em dias
        }

        input_df = pd.DataFrame(data, index=[0])

        # Realizar a previsão usando o modelo carregado
        prediction = prever_com_threshold(model, input_df, threshold_online)

        # Exibir o resultado da previsão com caixa colorida ao redor do texto
        if prediction == 1:
            st.subheader('Resultado da Previsão')
            st.markdown('<div style="border:2px solid green; border-radius: 5px; padding: 10px; color: green;">'
                        'O cliente provavelmente responderá positivamente.</div>', unsafe_allow_html=True)
        else:
            st.subheader('Resultado da Previsão')
            st.markdown('<div style="border:2px solid red; border-radius: 5px; padding: 10px; color: red;">'
                        'O cliente provavelmente não responderá positivamente.</div>', unsafe_allow_html=True)

# Seção Analytics
elif tabs == 'Analytics':
    threshold_analytics = st.sidebar.slider("Escolha do Threshold:", 0.0, 1.0, 0.5, step=0.05)

    st.subheader('Analytics')

    # Carregar arquivo CSV
    uploaded_file = st.file_uploader("Carregar arquivo CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Realizar a previsão usando o modelo carregado
        df['Prediction'] = prever_com_threshold(model, df, threshold_analytics)

        # Remover a formatação que altera as cores na tabela
        st.dataframe(df)

        # Gráficos de distribuição das features
        st.subheader('Distribuição das Features por Previsão')

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        sns.histplot(df, x='Income', hue='Prediction', multiple='stack', kde=True, ax=axs[0, 0])
        axs[0, 0].set_title('Distribuição de Income')
        axs[0, 0].set_xlabel('Income')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].legend(['Prediction 0', 'Prediction 1'])

        sns.histplot(df, x='Recency', hue='Prediction', multiple='stack', kde=True, ax=axs[0, 1])
        axs[0, 1].set_title('Distribuição de Recency')
        axs[0, 1].set_xlabel('Recency')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].legend(['Prediction 0', 'Prediction 1'])

        sns.histplot(df, x='NumCatalogPurchases', hue='Prediction', multiple='stack', kde=True, ax=axs[1, 0])
        axs[1, 0].set_title('Distribuição de NumCatalogPurchases')
        axs[1, 0].set_xlabel('NumCatalogPurchases')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].legend(['Prediction 0', 'Prediction 1'])

        sns.histplot(df, x='NumWebVisitsMonth', hue='Prediction', multiple='stack', kde=True, ax=axs[1, 1])
        axs[1, 1].set_title('Distribuição de NumWebVisitsMonth')
        axs[1, 1].set_xlabel('NumWebVisitsMonth')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].legend(['Prediction 0', 'Prediction 1'])

        plt.tight_layout()
        st.pyplot(fig)
