
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

#Sugestao do Code Academy
np.set_printoptions(suppress=True, precision = 2)
# %%
#Realizando a leitura dos dados e printando as primeiras linhas
nba = pd.read_csv('nba_games.csv')
nba.head()
# %%
# Filtrando os dados dos anos 2010 e 2014
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]
# %%
# Filtrar os pontos dos Knicks 2010
knicks_pts = nba_2010[nba_2010["fran_id"] == "Knicks"]["pts"]
# Filtrar os pontos dos Knicks 2014
knicks_pts_14 = nba_2014[nba_2014["fran_id"] == "Knicks"]["pts"]

# Filtrar os pontos dos Nets 2010
net_pts = nba_2010[nba_2010["fran_id"] == "Nets"]["pts"]
# Filtrar os pontos dos Nets 2014
net_pts_14 = nba_2014[nba_2014["fran_id"] == "Nets"]["pts"]

# %%
#Verificando
print(knicks_pts)
print(net_pts)
# %%
#Diferença entre os times no ano de 2010
knicks_avg_pts = nba_2010[nba_2010["fran_id"] == "Knicks"]["pts"].mean()
nets_avg_pts = nba_2010[nba_2010["fran_id"] == "Nets"]["pts"].mean()
diff_means_2010 = knicks_avg_pts - nets_avg_pts
#Diferença entre os times no ano de 2014
knicks_avg_pts_14 = nba_2014[nba_2014["fran_id"] == "Knicks"]["pts"].mean()
nets_avg_pts_14 = nba_2014[nba_2014["fran_id"] == "Nets"]["pts"].mean()
diff_means_2014 = knicks_avg_pts_14 - nets_avg_pts_14

#Diferença média em 2010
print(f"Point diference in 2010", knicks_avg_pts)
print(f"Point diference in 2010",nets_avg_pts)

#Diferença média em 2014
print(f"Point diference in 2014", knicks_avg_pts_14)
print(f"Point diference in 2014",nets_avg_pts_14)

#Printando o resultado
print(f"Difference in average points in 2010: {diff_means_2010}")
print(f"Difference in average points in 2014: {diff_means_2014}")
# %%
# Criando o histograma solicitado
plt.figure(figsize=(10, 6))
plt.hist(knicks_pts, bins=15, alpha=0.5, label='Knicks', color='blue')
plt.hist(net_pts, bins=15, alpha=0.5, label='Nets', color='orange')

# Adicionando titulo e legenda
plt.xlabel('Points Scored')
plt.ylabel('Frequency')
plt.title('Distribution of Points Scored: Knicks vs. Nets (2010)')
plt.legend()
plt.show()

# %%
#Criando um gráfico de caixa
sns.boxplot(data= nba_2010, x = "fran_id", y = "pts")
plt.xlabel('NBA Teams')
plt.ylabel('Points')
plt.show()
# %%
#Verificando a relação de resultado de jogo com jogos em casa e fora de casa
location_result_freq = pd.crosstab(nba_2010.game_location, nba_2010.game_result)
print(location_result_freq)
# %%
#Colocando em proporção %
location_result_prop = pd.crosstab(nba_2010.game_location, nba_2010.game_result, normalize=True)
print(location_result_prop)
# %%
#Frenquencia ao quadrado
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print(expected)
print(chi2)
# %%
#Calculando a correlação
forecast_cov = np.cov(nba_2010.point_diff, nba_2010.forecast)
print(forecast_cov)
# %%
point_diff_forecast_corr = pearsonr(nba_2010.forecast, nba_2010.point_diff)
point_diff_forecast_corr
# %%
#Criando um gráfico de bolinhas para melhor vizualição 
plt.scatter(data= nba_2010, x="forecast", y="point_diff")
plt.show()
plt.scatter(data= nba_2010, x="point_diff", y="forecast")
plt.show()
# %%
plt.clf() #to clear the previous plot
plt.scatter('forecast', 'point_diff', data=nba_2010)
plt.xlabel('Forecasted Win Prob.')
plt.ylabel('Point Differential')
plt.show()
# %%
