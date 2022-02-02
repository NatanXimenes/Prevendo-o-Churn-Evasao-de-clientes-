## **Apresentação do Case e sua solução**

![10](https://user-images.githubusercontent.com/88242628/152243195-e8591b18-d60b-4221-a265-3b830a1076cc.png)
# Previsão de Churn
Nesse repositório, estaremos explorando uma base de dados de uma empresa de serviços bancários com o intuito identificar o perfil dos clientes que deixaram a agência e prever a probabilidade de Churn.
- Churn.ipynb notebook com a solução do problema.
- Churn_Modelling.csv Dataset utilizado.
- Resultado_Modelo_NatanXimenes.csv Valores preditos pelo modelo.

# Contexto
Nesse repositório estaremos explorando uma base de dados de uma grande empresa de serviços bancários. Ela atua principalmente nos países da Europa oferecendo produtos financeiros, desde contas bancárias até investimentos, passando por alguns tipos de seguros e produto de investimento.

O modelo de negócio da empresa é do tipo serviço, ou seja, ela comercializa serviços bancários para seus clientes através de agências físicas e um portal online. 
Foi constatado uma alta taxa de clientes cancelando suas contas e deixando o banco e temos como o objetivo reduzir a evasão de clientes, ou seja, impedir que o cliente cancele seu contrato. Essa evasão, nas métricas de negócio, é conhecida como Churn.

Mas qual é a definição de churn? 
De maneira geral, Churn é uma métrica que indica o número de clientes que cancelaram o contrato ou pararam de comprar seu produto em um determinado período de tempo. Por exemplo, clientes que cancelaram o contrato de serviço ou após o vencimento do mesmo, não renovaram, são clientes considerados em churn.

## Estratégia a ser abordada
Tendo o problema e os dados em mãos, podemos estruturar uma solução da seguinte maneira:
- 1: Preparar os dados e em seguida entender o dataset a partir de vizualizações estatísticas
- 2: Direcionar o foco da nossa análise nos clientes que deixaram a empresa, qual a a sua faixa de idade, saldo bancário, localização, etc...
- 3: Criar um modelo de Machine Learning para prever quando um cliente provavelmente deixará de usar o serviço para que o time de negócio possa assim tomar medidas preventivas, afim de evitar essa saída.

## Nesse conjunto de dados temos as seguintes variáveis:
- CustomerId: identificação do cliente;
- Surname: sobrenome do cliente;
- CreditScore: pontuação de credito, 0 alto risco de inadimplência e 1000 clientes com baixo risco de inadimplência;
- Geography: país que o serviço é oferecido;
- Gender: sexo do cliente;
- Age: idade do cleinte;
- Tenure: um indicativo de estabilidade no emprego, em que 0 significa pouca estabilidade e 10 muita estabilidade.
- Balance: saldo da conta corrente;
- NumOfProducts: número de produtos bancários adquiridos;
- HasCrCard: se tem cartão de credito ou não, (Sim = 1 e Não = 0);
- IsActiveMember: se é um cliente com conta ativa, (Ativo = 1) ;
- EstimatedSalary: salário estimado;
- Exited: cliente deixou de ser cliente do banco ou não (Churn = 1).

# Análise exploratória dos dados
Com os dados coletados e limpos, podemos fazer uma análise expoloratória e tirar conclusões baseada nos dados:

![0](https://user-images.githubusercontent.com/88242628/152235880-7d1a9834-cc73-4617-b936-6beeb96df2f4.png)
**Observando a tabela que descreve os dados podemos dizer que os clientes dessa agência bancária possuem em *média*:**
* **39 anos**
* **Creditscore de 650 pontos**
* **Estabilidade de emprego de 5**
* **76 mil de saldo em sua conta e 1,5 produtos**
* **O salário estimado(EstimatedSalary) é de 100.000**
* **70% possuem cartão de crédito e 51% são membros ativos**
Histograma:
![1](https://user-images.githubusercontent.com/88242628/152234357-4ffb37c5-8e0b-4f0a-a40a-13eb5eb094ce.png)

- **a maioria dos clientes têm entre 30 a 45 anos e poucos clientes têm idade superior a 60 anos.**
- **a variável balance mostra uma inflação de clientes que têm pouco dinheiro na conta bancária, isso pode implicar que talvez esses clientes possuem outra conta bancária.**
- **muitos clientes possuem 1 ou 2 produtos, mas poucos possuem 3 ou mais.**
- **cerca de 2mil clientes deixaram o banco, vamos analisar isso**

Contagem de quem deixou o serviço(1) e quem manteve(0):

![2](https://user-images.githubusercontent.com/88242628/152234368-77668fc3-b00a-4f7c-91b9-db361e8eed07.png)

Percebemos que cerca de 20%(2 mil) dos clientes forma classificados como 1 (cancelaram a conta ou mudaram de banco). Portanto, o modelo de base pode prever que 20% dos clientes se desligarão. Dado que 20% é um número pequeno, precisamos garantir que o modelo escolhido preveja com grande precisão esses 20%, pois é do interesse do banco identificar e manter esse grupo, em vez de prever com precisão os clientes que serão retidos.

**Iremos posteriormente, lidar com esse desbalanceamento entre as classes.**

Avaliando as variáveis categóricas:

![3](https://user-images.githubusercontent.com/88242628/152237153-5a0b7762-f3a1-49ad-b117-d00f30bc2e00.png)

- **O numero de churn é maior entre as mulheres**
- **A maioria dos dados é de pessoas da França, mas uma proporção elevada de pessoas da alemanha deixaram o banco**
- **Curiosamente, a maioria dos clientes que mudaram são aqueles com cartões de crédito. Dado que a maioria dos clientes possui cartão de crédito poderá ser apenas uma coincidência.**
- **Não é surpresa que os membros inativos tenham uma rotatividade maior. O preocupante é que a proporção geral de clientes inativos é bastante alta, sugerindo que o banco pode precisar de um programa implementado para direcionar esse grupo para clientes ativos, pois isso definitivamente terá um impacto positivo na rotatividade de clientes.**

Avaliando as variáveis continuas comparando clientes que realizaram o churn com os que mantiveram o serviço:

![4](https://user-images.githubusercontent.com/88242628/152236972-524ac1ab-9eb0-46b9-b547-6c9761a07697.png)

![5](https://user-images.githubusercontent.com/88242628/152236978-8e81cddb-bf71-4d2d-9419-11938e457c69.png)

- **Com relação à estabilidade, os clientes nas duas extremidades (gastaram pouco tempo com o banco ou muito tempo com o banco) são mais propensos a desistir em comparação com aqueles que têm mandato médio.**
- **Os clientes mais velhos estão saindo mais do que os mais jovens e observa-se a presença de outliers (pontos discrepantes).**
- **É preocupante que o banco esteja perdendo clientes com saldos bancários significativos. Há uma maior variância na variável Tenure para os clientes que deixaram o banco.**
- **Não há diferença significativa na distribuição dos produtos e salários entre as classes, implicando que no geral não influenciam tanto.**
- **Não há tanta diferença significativa na distribuição da pontuação de crédito entre clientes retidos e cancelados, exceto que no box-plot dos clientes que cancelaram o serviço, pode-se observar outliers na parte inferior e um limite inferior menor do que o boxplot dos clientes que não cancelaram o serviço. Indicando assim, que clientes que cancelaram o serviço possuem um score menor do que os clientes que não cancelaram o serviço.**

Após a análise exploratória conseguimos definir o perfil dos clientes que deixaram o serviço e entender melhor os dados, a próxima etapa é preparar os dados para posteriormente aplicar modelos de Machine Learning para prever quais clientes podem deixar ou não o serviço futuramente.

# **Pré-Processamento dos dados**
Nessa etapa foi feita as seguintes transformações nos dados:
- 1) A transfomação das variáveis categóricas em binárias
- 2) Separação da base em treino e teste
- 3) O tratamento das classes desbalanceadas, será explicado logo abaixo
- 4) Deixar os dados na mesma escala com o uso do Standart Scaler

# 3) Utilizando o método SMOTE para lidar com os dados desbalanceados
Definição: Consiste em gerar dados sintéticos (não duplicados) da classe minoritária a partir de vizinhos. Foi aplicado a base de treino devido a grande diferença no número de clientes que realizaram o churn aos que mantiveram o serviço como contatado no início da análise.

![6](https://user-images.githubusercontent.com/88242628/152234409-44585c3e-e3c2-4bd2-9403-6e816a4dba4e.png)

Ao aplicar a técnica SMOTE, obtive os mesmo valores para as duas classe, foi constatado posteriormente que melhorou significamente a performace dos modelos

# **Treinando os modelos:**
Agora, nós podemos treinar modelos, ver como se saem e escolher o melhor a partir da medida de performace que é a ROC AUC que foi a escolhida por considerar a taxa de verdadeiros-positivos contra a taxa de falsos-positivos. Ou seja, numero de vezes que o classificador acertou a predição contra o número de vezes que o classificador errou a predição.

Serão treinados e avaliados 4 modelos de classificação: Decision tree, Logistic Regression, Random Forest e XGB Classifier com seus parâmetros padrões para avaliar os melhores e posteriormente usar o grid/random search para a escolha dos melhores parâmetros e avaliar melhor sua performace pela acurácia, recall e a matriz de confusão.

Tivemos os seguintes resultados
- roc_auc_score DecisionTreeClassifier =  0.7070520979094739 
- roc_auc_score LogisticRegression =  0.7146473448780551
- roc_auc_score RandomForestClassifier =  0.7467425622130185
- roc_auc_score XGBClassifier =  0.7596166113003533

**Os modelos que se sairam melhores foram o RandomForestClassifier e o XGBClassifier, diante disso com o uso do Grid/random search encontramos os melhores parâmetros para esses modelos, tivemos resultados bem semelhantes e uma boa técnica a ser implementada seria a combinação desses modelos com o uso do método Ensemble.**

# Método Ensemble
Com uso do RadomForest e o XGB já com seus melhores parâmetros, usaremos o Método Essemble que combina os modelos que melhorou ainda mais nosso resultado que foi o seguinte:

![7](https://user-images.githubusercontent.com/88242628/152235894-c86cd238-53e2-4228-8340-2d4623047a98.png)

Podemos notar pelas métricas de avaliação que o modelo preve muito bem quem não realiza o Churn, mas não tem a mesma eficiência ao prever quem realizou o Churn. Com este modelo, erramos em 204 clientes que saíram e não prevemos e 153 que ficaram e prevemos que saíram. Mesmo assim o resultado foi satisfatório diante dos dados disponibilizados. Para ter melhores resultados seriam necessário mais dados.

**Vamos estimar as variáveis mais importante para o Churn dos clientes:**

![8](https://user-images.githubusercontent.com/88242628/152234422-87d10980-53fd-461a-8294-bf47c5397537.png)

**foram elas: idade, saldo bancário, pontuação de crédito, estimativa de salário, membro ativo e país Alemanha.**

# Concluindo
A partir desse resultado, das características dos clientes que cancelaram o serviço, do modelo preditivo e com a informação das varíaveis predominates para ocasionar o churn, o time de negócio poderá prever clientes com a maior de cancelamento e assim tomar medidas preventivas, afim de evitar essa saída.
