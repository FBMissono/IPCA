import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics

st.set_page_config(page_title = 'IPCA - Predição da Inflação Mensal', \
        layout="wide",
        initial_sidebar_state='expanded'
    )


st.title('**PROJETO DE MODELO DE PREDIÇÃO DA INFLAÇÃO MENSAL PELO INDICE NACIONALDE PREÇOS AO CONSUMIDOR AMPLO (IPCA)**')
st.title(' ')
st.header('**I - DATASET extraído do site do SIDRA-IBGE (IPCA) - *carregamento dos dados***:')
st.markdown(' ')
st.markdown('''**IPCA mensal BR e setores- grupos e subgrupos**:''')
st.markdown('* https://sidra.ibge.gov.br/tabela/7060')
st.markdown('* https://sidra.ibge.gov.br/tabela/1420')

df = pd.read_excel('tabela7060.xlsx')
df = df.T.reset_index(drop=True).drop([0,1,31,32], axis=1)
df = df.drop([0], axis=0)
df.rename(columns={2:'Competencia', 3:'IPCA', 4:'Alimentacao_Domicilio', 5:'Alimentacao_Fora', 6:'Habitacao_Encargos_Manutenção', 7:'Habitacao_Combustivel_Energia',
        8: 'Residencia_Moveis_Utensilios', 9: 'Residencia_Aparelhos_Eletronicos', 10:'Residencia_Consertos_Manutencao', 11:'Vestuario_Roupas',
        12:'Vestuario_Calcados', 13:'Vesturario_Joias_Biju', 14:'Vestuario_Tecidos', 15:'Transporte_Público', 16:'Veiculo_Proprio', 17:'Combustiveis_veiculos', 
        18: 'Produtos_Farmaceuticos', 19:'Produtos_Oticos', 20:'Serv_Medicos_e_Dentarios', 21:'Serv_Laboratoriais_e_Hosp', 22:'Plano_Saude',
        23:'Higiene_Pessoal', 24:'Serv_Pessoais', 25:'Recreacao', 26:'Cursos_Regul', 27:'Leitura', 28:'Papelaria', 29:'Cursos_Diversos', 30:'Comunicacao'}, inplace = True)
df=df.drop([1], axis=0)

df['Competencia'] = df['Competencia'].str.replace(' ', '/')
df = df[['Competencia','Alimentacao_Domicilio', 'Alimentacao_Fora','Habitacao_Encargos_Manutenção', 'Habitacao_Combustivel_Energia',
       'Residencia_Moveis_Utensilios', 'Residencia_Aparelhos_Eletronicos', 'Residencia_Consertos_Manutencao', 'Vestuario_Roupas',
       'Vestuario_Calcados', 'Vesturario_Joias_Biju', 'Vestuario_Tecidos','Transporte_Público', 'Veiculo_Proprio', 'Combustiveis_veiculos',
       'Produtos_Farmaceuticos', 'Produtos_Oticos', 'Serv_Medicos_e_Dentarios','Serv_Laboratoriais_e_Hosp', 'Plano_Saude', 'Higiene_Pessoal',
       'Serv_Pessoais', 'Recreacao', 'Cursos_Regul', 'Leitura', 'Papelaria','Cursos_Diversos', 'Comunicacao', 'IPCA']]
df = df.reset_index(drop=True)

df1=pd.read_excel('tabela1420_1.xlsx')
df1 = df1.drop([0], axis=0)
df1.rename(columns={'Unnamed: 0':'Competencia', 'Unnamed: 1':'IPCA', 'Unnamed: 2':'Alimentacao_Domicilio', 'Unnamed: 3':'Alimentacao_Fora', 
                    'Unnamed: 4':'Habitacao_Encargos_Manutenção', 'Unnamed: 5':'Habitacao_Combustivel_Energia','Unnamed: 6': 'Residencia_Moveis_Utensilios',
                    'Unnamed: 7': 'Residencia_Aparelhos_Eletronicos', 'Unnamed: 8':'Residencia_Consertos_Manutencao', 'Unnamed: 9':'Vestuario_Roupas',
                    'Unnamed: 10':'Vestuario_Calcados', 'Unnamed: 11':'Vesturario_Joias_Biju', 'Unnamed: 12':'Vestuario_Tecidos','Unnamed: 13':'Transporte_Público', 
                    'Unnamed: 14':'Veiculo_Proprio', 'Unnamed: 15':'Combustiveis_veiculos', 'Unnamed: 16': 'Produtos_Farmaceuticos', 'Unnamed: 17':'Produtos_Oticos',
                    'Unnamed: 18':'Serv_Medicos_e_Dentarios', 'Unnamed: 19':'Serv_Laboratoriais_e_Hosp', 'Unnamed: 20':'Plano_Saude','Unnamed: 21':'Higiene_Pessoal',
                    'Unnamed: 22':'Serv_Pessoais', 'Unnamed: 23':'Recreacao', 'Unnamed: 24':'Cursos_Regul', 'Unnamed: 25':'Leitura', 'Unnamed: 26':'Papelaria',
                    'Unnamed: 27':'Cursos_Diversos', 'Unnamed: 28':'Comunicacao'}, inplace = True)
df1['Competencia'] = df1['Competencia'].str.replace(' ','/')
df1 = df1[['Competencia','Alimentacao_Domicilio', 'Alimentacao_Fora','Habitacao_Encargos_Manutenção', 'Habitacao_Combustivel_Energia',
       'Residencia_Moveis_Utensilios', 'Residencia_Aparelhos_Eletronicos', 'Residencia_Consertos_Manutencao', 'Vestuario_Roupas',
       'Vestuario_Calcados', 'Vesturario_Joias_Biju', 'Vestuario_Tecidos','Transporte_Público', 'Veiculo_Proprio', 'Combustiveis_veiculos',
       'Produtos_Farmaceuticos', 'Produtos_Oticos', 'Serv_Medicos_e_Dentarios','Serv_Laboratoriais_e_Hosp', 'Plano_Saude', 'Higiene_Pessoal',
       'Serv_Pessoais', 'Recreacao', 'Cursos_Regul', 'Leitura', 'Papelaria','Cursos_Diversos', 'Comunicacao', 'IPCA']]
df1 = df1.drop([1], axis =0)
df1 = df1.reset_index(drop=True)

df3 = pd.concat([df1,df], axis = 0, ignore_index=True)
df3

st.header('**II - DO IPCA**')
st.markdown(' ')
st.markdown('''O Índice Nacional de Preços ao Consumidor Amplo – IPCA consiste em um índice de inflação mensal medido através da inflação de um conjunto de produtos e
            serviços comercializados no varejo, referentes ao consumo pessoal das famílias. Esta faixa de renda foi criada com o objetivo de garantir uma cobertura de 90% das
            famílias pertencentes às áreas urbanas de cobertura do Sistema Nacional de Índices de Preços ao Consumidor - SNIPC.''')
st.markdown(' ')
st.markdown('Este conjunto de produtos e serviços comercializados no varejo é composto por 9 (nove) Grupos, quais sejam:')
st.markdown('* Alimentação e Bebidas;')
st.markdown('* Habitação;')
st.markdown('* Artigos de Residência;')
st.markdown('* Vestuário;')
st.markdown('* Transporte;')
st.markdown('* Saúde e Cuidados Pessoais;')
st.markdown('* Despesas Pessoais;')
st.markdown('* Educação;')
st.markdown('* Comunicação.')
st.markdown(' ')
st.markdown('OBS: cada um destes grupos encontram-se também subdivididos em subgrupos, conferindo maior especificidade ao conjunto de produtos e serviços.')
st.markdown(' ')
st.markdown('''Atualmente, a população-objetivo do IPCA abrange as famílias com rendimentos de 1 a 40 salários mínimos, qualquer que seja a fonte, residentes
             nas áreas urbanas das regiões de abrangência do SNIPC, as quais são: regiões metropolitanas de Belém, Fortaleza, Recife, Salvador, Belo Horizonte,
             Vitória, Rio de Janeiro, São Paulo, Curitiba, Porto Alegre, além do Distrito Federal e dos municípios de Goiânia, Campo Grande, Rio Branco, São Luís e Aracaju.''')

st.markdown('''|Colunas| Descrição |
|---|---|
|Competencia| Mês/Ano|
|Alimentacao_Domicilio| Taxa de inflação sobre os alimentos no domicílio (Cereais, Leguminosas, Oleaginosas, Farinhas, Massas, Açúcar, Hortalíças, Frutas, Carnes, Pescados, Carnes e Pescados Industrializados, Aves e ovos, etc) |
|Alimentacao_Fora| Taxa de inflação sobre alimentos fora do domicílio (Refeição, Lanche, Café da Manhã, Cafezinho, Cerveja, Doces)|
|Habitacao_Encargos_Manutenção| Taxa de Inflação do Subgrupo do grupo Habitação (Aluguel e taxas, Reparos, Artigos de Limpeza)|
|Habitacao_Combustivel_Energia|Taxa de Inflação do Subgrupo do grupo Habitação (Carvão Vegetal, Gás de botijão, Gás encanado, Energia Elétrica Residencial)|
|Residencia_Moveis_Utensilios|Taxa de Inflação do Subgrupo do Grupo Artigos Residenciais(mobiliário, tapete, cortina, cama, mesa e banho, etc.)|
|Residencia_Aparelhos_Eletronicos|Taxa de Inflação do Subgrupo do Grupo Artigos Residenciais|
|Recidencia_Consertos_Manutencao|Taxa de Inflação do Subgrupo do Grupo Artigos Residenciais|
|Vestuario_Roupas|Subgrupo de Vestuário (Taxa de Inflação)|
|Vestuario_Calcados|Subgrupo de Vestuário (Taxa de Inflação)|
|Vesturario_Joias_Biju| Subgrupo de Vestuário (Taxa de Inflação)|
|Vestuario_Tecidos| Subgrupo de Vestuário (Taxa de Inflação)|
|Transporte_Público|Subgrupo de Transporte (Taxa de Inflação)|
|Veiculo_Proprio| Subgrupo de Transporte (Taxa de Inflação)|
|Combustiveis_veiculos|Subgrupo de Transporte (Taxa de Inflação)|
|Produtos_Farmaceuticos|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Produtos_Oticos|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Serv_Medicos_e_Dentarios|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Serv_Laboratoriais_e_Hosp|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Plano_Saude|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Higiene_Pessoal|Subgrupo de Saúde e Cuidados Pessoais (Taxa de Inflação)|
|Serv_Pessoais|Subgrupo de Despesas Pessoais (Taxa de Inflação)|
|Recreacao|Subgrupo de Despesas Pessoais (Taxa de Inflação)|
|Cursos_Regul|Subgrupo de Educação (Taxa de Inflação)|
|Leitura|Subgrupo de Educação (Taxa de Inflação)|
|Papelaria|Subgrupo de Educação (Taxa de Inflação)|
|Cursos_Diversos|Subgrupo de Educação (Taxa de Inflação)|
|Comunicacao|Taxa de Inflação do Grupo Comunicação (Telefone Fixo, Celular, Internet, TV por Assinatura, etc)|
|IPCA| Target - índice mensal de inflação - âmbito Nacional  (Brasil)| ''')
st.markdown(' ')
st.header('**III - Tratamento de Dados Nulos e Tipos de Variáveis**')
st.markdown(' ')
df3[['Alimentacao_Domicilio', 'Alimentacao_Fora',
       'Habitacao_Encargos_Manutenção', 'Habitacao_Combustivel_Energia',
       'Residencia_Moveis_Utensilios', 'Residencia_Aparelhos_Eletronicos',
       'Residencia_Consertos_Manutencao', 'Vestuario_Roupas',
       'Vestuario_Calcados', 'Vesturario_Joias_Biju', 'Vestuario_Tecidos',
       'Transporte_Público', 'Veiculo_Proprio', 'Combustiveis_veiculos',
       'Produtos_Farmaceuticos', 'Produtos_Oticos', 'Serv_Medicos_e_Dentarios',
       'Serv_Laboratoriais_e_Hosp', 'Plano_Saude', 'Higiene_Pessoal',
       'Serv_Pessoais', 'Recreacao', 'Cursos_Regul', 'Leitura', 'Papelaria',
       'Cursos_Diversos', 'Comunicacao', 'IPCA']] = df3[['Alimentacao_Domicilio', 'Alimentacao_Fora',
       'Habitacao_Encargos_Manutenção', 'Habitacao_Combustivel_Energia',
       'Residencia_Moveis_Utensilios', 'Residencia_Aparelhos_Eletronicos',
       'Residencia_Consertos_Manutencao', 'Vestuario_Roupas',
       'Vestuario_Calcados', 'Vesturario_Joias_Biju', 'Vestuario_Tecidos',
       'Transporte_Público', 'Veiculo_Proprio', 'Combustiveis_veiculos',
       'Produtos_Farmaceuticos', 'Produtos_Oticos', 'Serv_Medicos_e_Dentarios',
       'Serv_Laboratoriais_e_Hosp', 'Plano_Saude', 'Higiene_Pessoal',
       'Serv_Pessoais', 'Recreacao', 'Cursos_Regul', 'Leitura', 'Papelaria',
       'Cursos_Diversos', 'Comunicacao', 'IPCA']].astype(float)
st.markdown('Não valores nulos a serem tratados. Foi transformado os valores do dataset em float, pois estavam todos com tipo objest')
st.subheader('**Entendimento dos Dados - Univariada**')
st.markdown(' ')
plt.rc('figure', figsize=(20, 22))
fig, axes = plt.subplots(7,4)

sns.histplot(ax = axes[0, 0], x='Alimentacao_Domicilio', data=df3)
sns.histplot(ax = axes[0, 1], x='Alimentacao_Fora', data=df3)
sns.histplot(ax = axes[0, 2], x='Habitacao_Encargos_Manutenção', data=df3)
sns.histplot(ax = axes[0, 3], x='Habitacao_Combustivel_Energia', data=df3)
sns.histplot(ax = axes[1, 0], x='Residencia_Moveis_Utensilios', data=df3)
sns.histplot(ax = axes[1, 1], x='Residencia_Aparelhos_Eletronicos', data=df3)
sns.histplot(ax = axes[1, 2], x='Residencia_Consertos_Manutencao', data=df3)
sns.histplot(ax = axes[1, 3], x='Vestuario_Roupas', data=df3)
sns.histplot(ax = axes[2, 0], x='Vestuario_Calcados', data=df3)
sns.histplot(ax = axes[2, 1], x='Vesturario_Joias_Biju', data=df3)
sns.histplot(ax = axes[2, 2], x='Vestuario_Tecidos', data=df3)
sns.histplot(ax = axes[2, 3], x='Transporte_Público', data=df3)
sns.histplot(ax = axes[3, 0], x='Veiculo_Proprio', data=df3)
sns.histplot(ax = axes[3, 1], x='Combustiveis_veiculos', data=df3)
sns.histplot(ax = axes[3, 2], x='Produtos_Farmaceuticos', data=df3)
sns.histplot(ax = axes[3, 3], x='Produtos_Oticos', data=df3)
sns.histplot(ax = axes[4, 0], x='Serv_Medicos_e_Dentarios', data=df3)
sns.histplot(ax = axes[4, 1], x='Serv_Laboratoriais_e_Hosp', data=df3)
sns.histplot(ax = axes[4, 2], x='Plano_Saude', data=df3)
sns.histplot(ax = axes[4, 3], x='Higiene_Pessoal', data=df3)
sns.histplot(ax = axes[5, 0], x='Serv_Pessoais', data=df3)
sns.histplot(ax = axes[5, 1], x='Recreacao', data=df3)
sns.histplot(ax = axes[5, 2], x='Cursos_Regul', data=df3)
sns.histplot(ax = axes[5, 3], x='Leitura', data=df3)
sns.histplot(ax = axes[6, 0], x='Papelaria', data=df3)
sns.histplot(ax = axes[6, 1], x='Cursos_Diversos', data=df3)
sns.histplot(ax = axes[6, 2], x='Comunicacao', data=df3)
sns.histplot(ax = axes[6, 3], x='IPCA', data=df3)

plt.subplots_adjust(wspace=0.15, hspace=0.45)
st.pyplot(plt)
st.subheader(' ')
st.subheader('**Entendimento dos Dados - Bivariada**')
plt.rc('figure', figsize=(20, 22))
fig, axes = plt.subplots(7,4)

axes[0,0].plot(df3['Alimentacao_Domicilio'], df3['IPCA'], '.')
axes[0,1].plot(df3['Alimentacao_Fora'], df3['IPCA'], '.')
axes[0,2].plot(df3['Habitacao_Encargos_Manutenção'], df3['IPCA'], '.')
axes[0,3].plot(df3['Habitacao_Combustivel_Energia'], df3['IPCA'], '.')
axes[1,0].plot(df3['Residencia_Moveis_Utensilios'], df3['IPCA'], '.')
axes[1,1].plot(df3['Residencia_Aparelhos_Eletronicos'], df3['IPCA'], '.')
axes[1,2].plot(df3['Residencia_Consertos_Manutencao'], df3['IPCA'], '.')
axes[1,3].plot(df3['Vestuario_Roupas'], df3['IPCA'], '.')
axes[2,0].plot(df3['Vestuario_Calcados'], df3['IPCA'], '.')
axes[2,1].plot(df3['Vesturario_Joias_Biju'], df3['IPCA'], '.')
axes[2,2].plot(df3['Vestuario_Tecidos'], df3['IPCA'], '.')
axes[2,3].plot(df3['Transporte_Público'], df3['IPCA'], '.')
axes[3,0].plot(df3['Veiculo_Proprio'], df3['IPCA'], '.')
axes[3,1].plot(df3['Combustiveis_veiculos'], df3['IPCA'], '.')
axes[3,2].plot(df3['Produtos_Farmaceuticos'], df3['IPCA'], '.')
axes[3,3].plot(df3['Produtos_Oticos'], df3['IPCA'], '.')
axes[4,0].plot(df3['Serv_Medicos_e_Dentarios'], df3['IPCA'], '.')
axes[4,1].plot(df3['Serv_Laboratoriais_e_Hosp'], df3['IPCA'], '.')
axes[4,2].plot(df3['Plano_Saude'], df3['IPCA'], '.')
axes[4,3].plot(df3['Higiene_Pessoal'], df3['IPCA'], '.')
axes[5,0].plot(df3['Serv_Pessoais'], df3['IPCA'], '.')
axes[5,1].plot(df3['Recreacao'], df3['IPCA'], '.')
axes[5,2].plot(df3['Cursos_Regul'], df3['IPCA'], '.')
axes[5,3].plot(df3['Leitura'], df3['IPCA'], '.')
axes[6,0].plot(df3['Papelaria'], df3['IPCA'], '.')
axes[6,1].plot(df3['Cursos_Diversos'], df3['IPCA'], '.')
axes[6,2].plot(df3['Comunicacao'], df3['IPCA'], '.')
axes[6,3].plot(df3['IPCA'], df3['IPCA'], '.')


axes[0,0].set_xlabel('Alimentacao_Domicilio')
axes[0,0].set_ylabel('IPCA')
axes[0,1].set_xlabel('Alimentacao_Fora')
axes[0,2].set_xlabel('Habitacao_Encargos_Manutenção')
axes[0,3].set_xlabel('Habitacao_Combustivel_Energia')
axes[1,0].set_xlabel('Residencia_Moveis_Utensilios')
axes[1,0].set_ylabel('IPCA')
axes[1,1].set_xlabel('Residencia_Aparelhos_Eletronicos')
axes[1,2].set_xlabel('Residencia_Consertos_Manutencao')
axes[1,3].set_xlabel('Vestuario_Roupas')
axes[2,0].set_xlabel('Vestuario_Calcados')
axes[2,0].set_ylabel('IPCA')
axes[2,1].set_xlabel('Vesturario_Joias_Biju')
axes[2,2].set_xlabel('Vestuario_Tecidos')
axes[2,3].set_xlabel('Transporte_Público')
axes[3,0].set_xlabel('Veiculo_Proprio')
axes[3,0].set_ylabel('IPCA')
axes[3,1].set_xlabel('Combustiveis_veiculos')
axes[3,2].set_xlabel('Produtos_Farmaceuticos')
axes[3,3].set_xlabel('Produtos_Oticos')
axes[4,0].set_xlabel('Serv_Medicos_e_Dentarios')
axes[4,0].set_ylabel('IPCA')
axes[4,1].set_xlabel('Serv_Laboratoriais_e_Hosp')
axes[4,2].set_xlabel('Plano_Saude')
axes[4,3].set_xlabel('Higiene_Pessoal')
axes[5,0].set_xlabel('Serv_Pessoais')
axes[5,0].set_ylabel('IPCA')
axes[5,1].set_xlabel('Recreacao')
axes[5,2].set_xlabel('Cursos_Regul')
axes[5,3].set_xlabel('Leitura')
axes[6,0].set_xlabel('Papelaria')
axes[6,0].set_ylabel('IPCA')
axes[6,1].set_xlabel('Cursos_Diversos')
axes[6,2].set_xlabel('Comunicacao')
axes[6,3].set_xlabel('IPCA')

plt.subplots_adjust(wspace=0.15, hspace=0.45)
st.pyplot(plt)
st.subheader('')
st.subheader('**Tratamento de Outoliers**')
st.subheader('')
st.markdown('> Considerando que todas as variáveis são taxas de inflação em determinados setores ou grupos, o que lhes atriubuiem a característica de variáveis quantitativas contínuas, entendemos que não há outliers a serem tratadas. Ademias, a maiore variação de valores se encontram na variável ***Habitação - Combustível e eneuergia***, cujos valores encontram-se entre -6 a 8.')
st.header('')
st.header('**IV - Da Modelagem de Previsão de Inflação pelo IPCA**')
st.subheader('')
st.subheader('**Da Divisão do Dataset em Base de treino e Base de Teste, na proporção de 75% e 25%, respectivamente**')
st.subheader('')
df4 = df3.drop(columns=['Competencia'])
X = df4.loc[:, 'Alimentacao_Domicilio':'Comunicacao']
y = df4[['IPCA']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, train_size = 0.75, random_state=1000)
st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, train_size = 0.75, random_state=1000)')
st.markdown(' ')
st.markdown('> Considerando que o Dataset possui somente 86 linhas. Dividimos o dataset somente em Treino e Teste, até para se manter um numero de linhas consideráveis para a aprendizagem da inteligência artifical do modelo preditivo. Enfim não adotamos o uma base de fora do treino e treste (Out of Time - OOT)')
st.subheader('')
st.subheader('**Rodando Modelo pelo Metodo RandomFlorestRegressor**')
clf = RandomForestRegressor(n_estimators=10)
clf.fit(X_train,y_train.values.ravel())
st.code('''clf = RandomForestRegressor(n_estimators=10)
clf.fit(X_train,y_train.values.ravel())''')

y_pred = clf.predict(X_test)
df_acertos = y_test.copy()
df_acertos['IPCA_preditado'] = y_pred
df_acertos['Diferença'] = df_acertos.IPCA - df_acertos.IPCA_preditado
df_acertos.loc[(df_acertos['Diferença']<=0.05) & (df_acertos['Diferença']>= -0.05), 'Acerto']= 1
df_acertos = df_acertos.fillna(0)
df_acertos

df_acuracia = pd.DataFrame(df_acertos.Acerto.value_counts())
df_acuracia= df_acuracia.reset_index(drop=False)
df_acuracia

Acuracia = df_acuracia.iloc[1,1]/(df_acuracia.iloc[0,1]+df_acuracia.iloc[1,1]) 
st.markdown(f'Acurácia de {Acuracia}.')
st.subheader('')
st.subheader('**Rodando Modelo Statsmodels**')
st.subheader(' ')
df5 = pd.concat([X_train, y_train], axis=1)
df5_test = pd.concat([X_test, y_test], axis=1)
results = smf.ols('''IPCA ~ Alimentacao_Domicilio +
                  Alimentacao_Fora + 
                  Habitacao_Encargos_Manutenção +
                  Habitacao_Combustivel_Energia +
                  Residencia_Moveis_Utensilios +
                  Residencia_Aparelhos_Eletronicos +
                  Residencia_Consertos_Manutencao + 
                  Vestuario_Roupas +
                  Vestuario_Calcados + 
                  Vesturario_Joias_Biju + 
                  Vestuario_Tecidos +
                  Transporte_Público + 
                  Veiculo_Proprio + 
                  Combustiveis_veiculos +
                  Serv_Laboratoriais_e_Hosp +
                  Plano_Saude +
                  Higiene_Pessoal +
                  Serv_Pessoais +
                  Recreacao +
                  Cursos_Regul +
                  Leitura +
                  Papelaria +
                  Cursos_Diversos +
                  Comunicacao''', 
                  data=df5).fit()
st.code('''results = smf.ols(IPCA ~ Alimentacao_Domicilio +
                  Alimentacao_Fora + 
                  Habitacao_Encargos_Manutenção +
                  Habitacao_Combustivel_Energia +
                  Residencia_Moveis_Utensilios +
                  Residencia_Aparelhos_Eletronicos +
                  Residencia_Consertos_Manutencao + 
                  Vestuario_Roupas +
                  Vestuario_Calcados + 
                  Vesturario_Joias_Biju + 
                  Vestuario_Tecidos +
                  Transporte_Público + 
                  Veiculo_Proprio + 
                  Combustiveis_veiculos +
                  Serv_Laboratoriais_e_Hosp +
                  Plano_Saude +
                  Higiene_Pessoal +
                  Serv_Pessoais +
                  Recreacao +
                  Cursos_Regul +
                  Leitura +
                  Papelaria +
                  Cursos_Diversos +
                  Comunicacao, 
                  data=df5).fit()''')
st.write(results.summary())
st.markdown('')
st.markdown('As variáveis irrelevantes são aquelas com P>|t| maiores de 0,5, ou seja, são irrelevantes as variáveis:')
st.markdown('* Residencia_Consertos_Manutencao;')
st.markdown('* Vestuario_Calcados;')
st.markdown('* Serv_Laboratoriais_e_Hosp;')
st.markdown('* Serv_Pessoais;')
st.markdown('* Recreacao;')
st.markdown('* Comunicacao')
st.markdown('')
st.code('''results = smf.ols(IPCA ~ Alimentacao_Domicilio +
                  Alimentacao_Fora + 
                  Habitacao_Encargos_Manutenção +
                  Habitacao_Combustivel_Energia +
                  Residencia_Moveis_Utensilios +
                  Residencia_Aparelhos_Eletronicos +
                  Vestuario_Roupas +
                  Vesturario_Joias_Biju + 
                  Vestuario_Tecidos +
                  Transporte_Público + 
                  Veiculo_Proprio + 
                  Combustiveis_veiculos +
                  Plano_Saude +
                  Higiene_Pessoal +
                  Serv_Pessoais +
                  Cursos_Regul +
                  Leitura +
                  Papelaria +
                  Cursos_Diversos, 
                  data=df5).fit()
''')
results = smf.ols('''IPCA ~ Alimentacao_Domicilio +
                  Alimentacao_Fora + 
                  Habitacao_Encargos_Manutenção +
                  Habitacao_Combustivel_Energia +
                  Residencia_Moveis_Utensilios +
                  Residencia_Aparelhos_Eletronicos +
                  Vestuario_Roupas +
                  Vesturario_Joias_Biju + 
                  Vestuario_Tecidos +
                  Transporte_Público + 
                  Veiculo_Proprio + 
                  Combustiveis_veiculos +
                  Plano_Saude +
                  Higiene_Pessoal +
                  Serv_Pessoais +
                  Cursos_Regul +
                  Leitura +
                  Papelaria +
                  Cursos_Diversos''', 
                  data=df5).fit()
st.write(results.summary())
results = smf.ols('''IPCA ~ Alimentacao_Domicilio +
                  Alimentacao_Fora + 
                  Habitacao_Encargos_Manutenção +
                  Habitacao_Combustivel_Energia +
                  Residencia_Moveis_Utensilios +
                  Residencia_Aparelhos_Eletronicos +
                  Vestuario_Roupas +
                  Vesturario_Joias_Biju + 
                  Vestuario_Tecidos +
                  Transporte_Público + 
                  Veiculo_Proprio + 
                  Combustiveis_veiculos +
                  Plano_Saude +
                  Higiene_Pessoal +
                  Serv_Pessoais +
                  Cursos_Regul +
                  Leitura +
                  Papelaria +
                  Cursos_Diversos''', 
                  data=df5_test).fit()
st.subheader('')
st.markdown('Fitando este modelo de regressão com a base teste, obtemos a acurácia de:')
st.write(results.rsquared)
st.markdown('')
st.markdown('Houve Melhora no modelo, visto que não houve uma alteração significativa da acurácia, mesmo com redução de 5 variáveis (irrelevantes).')
st.subheader(' ')
st.header('**V - Conclusão**')
st.markdown('''Verificamos que o melhor modelo preditivo foi com a utilização do Statsmodels Formula (OLS - Ordinary Least Squares), 
            em relação ao RandomFlorestRegressor. Ademais, verificamos que as variáveis mais irrelevantes foram:''')
st.markdown('* Residencia_Consertos_Manutencao;')
st.markdown('* Residencia_Consertos_Manutencao;')
st.markdown('* Vestuario_Calcados;')
st.markdown('* Serv_Laboratoriais_e_Hosp;')
st.markdown('* Serv_Pessoais;')
st.markdown('* Recreacao;')
st.markdown('* Comunicacao.')
st.subheader(' ')
st.markdown('Estas variáveis foram excluídas do modelo final.')
st.markdown('''Cabe esclarecer que este modelos ainda pode ser melhorado, 
            no sentido de se buscar o melhor preditivo com menor números de variáveis, desta forma poderá obter insight em relação ao grupo que mais impacta na taxa IPCA. ''')
st.markdown('''Enfim, trata-se de um trabalho inicial, sem pretensão de 
            exaurir o estudo, haja visto que este projeto poderá ser ainda desenvolvido com a inclusão de mais competências (mes/ano) no Dataset e novos insights a serem buscados.''')
