from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import explained_variance_score,mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

str="Afrikaans-AfriBooms&1315&425&0.360&0.094&85.47|Ancient\_Greek-Perseus&11476&1306&0.789&0.478&79.39|Ancient\_Greek-PROIEL&15014&1047&0.880&0.634&79.25|Bulgarian-BTB&8907&1116&0.861&0.544&91.22|Basque-BDT&5396&1799&0.825&0.513&19.53|Chinese-GSD&3997&500&0.524&0.513&76.77|Danish-DDT&4383&565&0.655&0.377&86.28|English-GUM&3753&890&0.631&0.381&85.05|English-LinES&3176&1035&0.680&0.401&81.97|Estonian-EDT&24633&3214&0.847&0.566&85.35|Irish-IDT&858&454&0.425&0.273&70.88|Gothic-PROIEL&3387&1029&0.876&0.690&69.55|Hindi-HDTB&13304&1684&0.760&0.296&92.41|Croatian-SET&6914&1136&0.651&0.212&87.36|Hungarian-Szeged&910&449&0.261&0.096&82.66|Indonesian-GSD&4477&557&0.657&0.257&80.05|Japanese-GSD&7125&550&0.585&0.275&83.11|Korean-GSD&4400&989&0.800&0.627&85.14|Korean-Kaist&23010&2287&0.972&0.614&86.91|Latin-ITTB&16809&2101&0.881&0.591&87.08|Latin-PROIEL&15917&1260&0.901&0.696&73.61|Norwegian-Bokmaal&15696&1939&0.851&0.534&91.23|Norwegian-Nynorsk&14174&1511&0.809&0.475&90.99|Norwegian-NynorskLIA&3412&957&0.809&0.638&70.34|Old\_Church\_Slavonic-PROIEL&4124&1141&0.911&0.765&75.73|Polish-LFG&13774&1727&0.981&0.931&94.86|Romanian-RRT&8043&729&0.713&0.158&86.87|Russian-Taiga&1435&884&0.743&0.537&74.24|Serbian-SET&3328&520&0.531&0.150&88.66|Slovak-SNK&8483&1061&0.855&0.562&88.85|Slovenian-SSJ&6478&788&0.731&0.385&91.47|Swedish-LinES&3176&1035&0.701&0.411&84.08|yghur-UDT&1656&900&0.810&0.557&67.05|Urdu-UDTB&4043&535&0.422&0.108&83.39"

X_prio,X_size,X_iso,X_both=[],[],[],[]
A_size,A_iso=[],[]
Y=[]
lines=str.split("|")
for line in lines: 
    line=line.split("&")
    X_prio.append([0])
    X_size.append([float(line[1])])
    A_size.append(float(line[1]))
    X_iso.append([float(line[-2])])
    A_iso.append(float(line[-2])/float(line[1]))
    X_both.append([float(line[1]),float(line[-2])])
    Y.append(float(line[-1])/100)

print("r-size:",pearsonr(A_size,Y))
print("r-isom:",pearsonr(A_iso,Y))

X_prio=normalize(X_prio)
X_size=normalize(X_size)
X_iso=normalize(X_iso)
X_both=normalize(X_both)


regressor = DecisionTreeRegressor(random_state=0)
regressor = LinearRegression()

print("Prio:",cross_val_score(regressor, X_prio, Y, cv=10,scoring=make_scorer(mean_absolute_error)).mean())
print("Size:",cross_val_score(regressor, X_size, Y, cv=10,scoring=make_scorer(mean_absolute_error)).mean())
print("Isom:",cross_val_score(regressor, X_iso, Y, cv=10,scoring=make_scorer(mean_absolute_error)).mean())
print("Both:",cross_val_score(regressor, X_both, Y, cv=10,scoring=make_scorer(mean_absolute_error)).mean())
