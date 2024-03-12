#!/usr/bin/env python
# coding: utf-8

# ### Librairies

# In[1]:


import pandas as pad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab


# ### Analyse de la base de données 

# In[2]:


# Lecture du fichier Excel 
df = pad.read_excel('Database_analyse.xlsx', header = 0)

print('La base de données comporte', len(df), 'patients.')


# ##### Sexe :

# In[3]:


print('Pourcentage hommes et femmes :\n')
print((df.value_counts('Sexe')*100)/len(df))


# In[4]:


x = (df.value_counts('Sexe')*100)/len(df)

plt.pie(x, colors = ['rebeccapurple', 'royalblue'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.5, labeldistance = 1.1)

plt.title('Diagramme de pourcentage : Hommes et Femmes', fontsize = 10)
plt.legend(['Hommes', 'Femmes'], bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figure(figsize = (4, 4))
#plt.savefig('Diagramme_HF.png', dpi=400)


# In[5]:


## DataFrame femme :

df_femme =  df[df['Sexe'] == 'F']


# In[6]:


## DataFrame homme :

df_homme =  df[df['Sexe'] == 'H']


# ##### Age au moment du traitement (années) :

# In[7]:


print('Age au moment du traitement (années) valeur minimal :', round(df['Age traitement (années)'].min(),2))
print('Age au moment du traitement (années) valeur maximal :', round(df['Age traitement (années)'].max(), 2))
print('Age au moment du traitement (années) valeur moyenne :', round(df['Age traitement (années)'].mean(),2))
print('Age au moment du traitement (années) valeur ecart-type :', round(df['Age traitement (années)'].std(),2))


# In[12]:


sns.histplot(data = df, x = df['Age traitement (années)'], kde = True, color = 'blue')


# In[9]:


age_traitement_femme = df_femme['Age traitement (années)']
age_traitement_homme = df_homme['Age traitement (années)']
 
boxplot_age  = [age_traitement_femme, age_traitement_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_age)
plt.title('Age des patients au moment du traitement (années)')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### Taille (m):

# In[10]:


print('Taille (m) valeur minimal :', round(df['Taille'].min(),2))
print('Taille (m) valeur maximal :', round(df['Taille'].max(), 2))
print('Taille (m) valeur moyenne :', round(df['Taille'].mean(),2))
print('Taille (m) valeur ecart-type :', round(df['Taille'].std(),2))


# In[13]:


sns.histplot(data = df, x = df['Taille'], kde = True, color = 'green')


# In[19]:


taille_femme = df_femme['Taille']
taille_homme = df_homme['Taille']
 
boxplot_taille  = [taille_femme, taille_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_taille)
plt.title('Taille des patients (m)')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### Poids (kg) :

# In[180]:


print('Poids (kg) valeur minimal :', round(df['Poids'].min(),2))
print('Poids (kg) valeur maximal :', round(df['Poids'].max(), 2))
print('Poids (kg) valeur moyenne :', round(df['Poids'].mean(),2))
print('Poids (kg) valeur ecart-type :', round(df['Poids'].std(),2))


# In[18]:


sns.histplot(data = df, x = df['Poids'], kde = True, color = 'orange')


# In[20]:


poids_femme = df_femme['Poids']
poids_homme = df_homme['Poids']
 
boxplot_poids  = [poids_femme, poids_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_poids)
plt.title('Poids des patients (kg)')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### IMC :

# In[181]:


print('IMC valeur minimal :', round(df['IMC'].min(),2))
print('IMC valeur maximal :', round(df['IMC'].max(), 2))
print('IMC valeur moyenne :', round(df['IMC'].mean(),2))
print('IMC valeur ecart-type :', round(df['IMC'].std(),2))


# In[21]:


sns.histplot(data = df, x = df['IMC'], kde = True, color = 'purple')


# In[22]:


imc_femme = df_femme['Poids']
imc_homme = df_homme['Poids']
 
boxplot_poids  = [imc_femme, imc_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_poids)
plt.title('IMC des patients')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### Décès :

# In[182]:


print('Pourcentage de décès :\n')

print((df.value_counts('Décès')*100)/len(df))


# In[183]:


x = (df.value_counts('Décès')*100)/len(df)

plt.pie(x, colors = ['rebeccapurple', 'royalblue', 'crimson'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.5, labeldistance = 1.1)

plt.title('Diagramme de pourcentage homme et femme : décès, perdu de vue, non décédé', fontsize = 10)
plt.legend(['Décédé', 'Perdu de vue', 'Non décédé'], bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figure(figsize = (4, 4))
#plt.savefig('Diagramme_deces.png', dpi=400)


# In[184]:


x = (df_femme.value_counts('Décès')*100)/len(df)

plt.pie(x, colors = ['rebeccapurple', 'royalblue', 'crimson'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.5, labeldistance = 1.1)

plt.title('Diagramme de pourcentage femme : décès, perdu de vue, non décédé', fontsize = 10)
plt.legend(['Décédé', 'Perdu de vue', 'Non décédé'], bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figure(figsize = (4, 4))
#plt.savefig('Diagramme_deces.png', dpi=400)


# In[23]:


x = (df_homme.value_counts('Décès')*100)/len(df)

plt.pie(x, colors = ['rebeccapurple', 'royalblue', 'crimson'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.5, labeldistance = 1.1)

plt.title('Diagramme de pourcentage homme : décès, perdu de vue, non décédé', fontsize = 10)
plt.legend(['Décédé', 'Perdu de vue', 'Non décédé'], bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figure(figsize = (4, 4))
#plt.savefig('Diagramme_deces.png', dpi=400)


# ###### Age du patient au moment du décès :

# In[186]:


print('Age au moment du décès (années) valeur minimal :', round(df['Age décès (années)'].min(),2))
print('Age au moment du décès (années) valeur maximal :', round(df['Age décès (années)'].max(), 2))
print('Age au moment du décès (années) valeur moyenne :', round(df['Age décès (années)'].mean(),2))
print('Age au moment du décès (années) valeur ecart-type :', round(df['Age décès (années)'].std(),2))


# In[24]:


sns.histplot(data = df, x = df['Age décès (années)'], kde = True, color = 'green')


# In[25]:


age_deces_femme = []
for x in df_femme['Age décès (années)']:
    if str(x).lower() != "nan":
        age_deces_femme.append(x) 
        
age_deces_homme = []
for x in df_homme['Age décès (années)']:
    if str(x).lower() != "nan":
        age_deces_homme.append(x) 

boxplot_age_deces  = [age_deces_femme, age_deces_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_age_deces)
plt.title('Age des patients au moment du décès (années)')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### Survie du patient :

# In[208]:


print('Survie du patient (années) valeur minimal :', round(df['Survie (années)'].min(),2))
print('Survie du patient (années) valeur maximal :', round(df['Survie (années)'].max(), 2))
print('Survie du patient (années) valeur moyenne :', round(df['Survie (années)'].mean(),2))
print('Survie du patient (années) valeur ecart-type :', round(df['Survie (années)'].std(),2))


# In[61]:


sns.histplot(data = df, x = df['Survie (années)'], kde = True, color = 'red')


# In[210]:


survie_femme = []
for x in df_femme['Survie (années)']:
    if str(x).lower() != "nan":
        age_deces_femme.append(x) 
        
survie_homme = []
for x in df_homme['Survie (années)']:
    if str(x).lower() != "nan":
        age_deces_homme.append(x) 

boxplot_survie  = [survie_femme, survie_homme]
boxplot_name = ['Femmes', 'Hommes']

plt.boxplot(boxplot_age)
plt.title('Survie des patients (années)')
pylab.xticks([1,2], boxplot_name)
plt.show


# ###### Temps entre l'IRM pré traitement et le traitement :

# In[30]:


print('Temps entre IRM pré traitement et traitement valeur minimal :', round(df['Temps IRM pré et traitement (jours)'].min(),2))
print('Temps entre IRM pré traitement et traitement valeur maximal :', round(df['Temps IRM pré et traitement (jours)'].max(), 2))
print('Temps entre IRM pré traitement et traitement valeur moyenne :', round(df['Temps IRM pré et traitement (jours)'].mean(),2))
print('Temps entre IRM pré traitement et traitement valeur ecart-type :', round(df['Temps IRM pré et traitement (jours)'].std(),2))


# In[29]:


sns.histplot(data = df, x = df['Temps IRM pré et traitement (jours)'], kde = True, color = 'pink')


# In[36]:


temps_pre_traitement = []

for x in df['Temps IRM pré et traitement (jours)']:
    if str(x).lower() != "nan":
        temps_pre_traitement.append(x) 

boxplot_pre_traitement  = [temps_pre_traitement]

plt.boxplot(boxplot_pre_traitement)
plt.title('Temps entre IRM pré traitement et traitement (jours)')
pylab.xticks([1,1])
plt.show


# ###### Temps entre le traitement et l'IRM post traitement :

# In[37]:


print('Temps entre traitement et IRM post traitement valeur minimal :', round(df['Temps traitement et IRM post (jours)'].min(),2))
print('Temps entre traitement et IRM post traitement valeur maximal :', round(df['Temps traitement et IRM post (jours)'].max(), 2))
print('Temps entre traitement et IRM post traitement valeur moyenne :', round(df['Temps traitement et IRM post (jours)'].mean(),2))
print('Temps entre traitement et IRM post traitement valeur ecart-type :', round(df['Temps traitement et IRM post (jours)'].std(),2))


# In[38]:


sns.histplot(data = df, x = df['Temps traitement et IRM post (jours)'], kde = True, color = 'cyan')


# In[39]:


temps_traitement_post = []

for x in df['Temps traitement et IRM post (jours)']:
    if str(x).lower() != "nan":
        temps_pre_traitement.append(x) 

boxplot_traitement_post  = [temps_traitement_post]

plt.boxplot(boxplot_pre_traitement)
plt.title('Temps entre traitement et IRM post traitement (jours)')
pylab.xticks([1,1])
plt.show


# ###### Machine de traitement :

# In[34]:


print('Pourcentage de machine RT :\n')

print((df.value_counts('Machine')*100)/len(df))


# In[47]:


x = (df.value_counts('Machine')*100)/len(df)

plt.pie(x, colors = ['rebeccapurple', 'royalblue', 'crimson'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.5, labeldistance = 1.1)

plt.title('Diagramme de pourcentage des machines de RT', fontsize = 10)
plt.legend(df['Machine'], bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figure(figsize = (4, 4))
#plt.savefig('Diagramme_deces.png', dpi=400)


# ###### Nombre de fractions :

# In[35]:


print('Nombre de fractions valeur minimal :', round(df['Fractions'].min(),2))
print('Nombre de fractions valeur maximal :', round(df['Fractions'].max(), 2))
print('Nombre de fractions valeur moyenne :', round(df['Fractions'].mean(),2))
print('Nombre de fractions valeur ecart-type :', round(df['Fractions'].std(),2))


# ###### Dose totale (Gy) :

# In[36]:


print('Dose totale (Gy) valeur minimal :', round(df['Dose totale (Gy)'].min(),2))
print('Dose totale (Gy) valeur maximal :', round(df['Dose totale (Gy)'].max(), 2))
print('Dose totale (Gy) valeur moyenne :', round(df['Dose totale (Gy)'].mean(),2))
print('Dose totale (Gy) valeur ecart-type :', round(df['Dose totale (Gy)'].std(),2))


# ###### Dose par fraction (Gy) :

# In[37]:


print('Dose par fraction (Gy) valeur minimal :', round(df['Dose/fraction (Gy)'].min(),2))
print('Dose par fraction (Gy) valeur maximal :', round(df['Dose/fraction (Gy)'].max(), 2))
print('Dose par fraction (Gy) valeur moyenne :', round(df['Dose/fraction (Gy)'].mean(),2))
print('Dose par fraction (Gy) valeur ecart-type :', round(df['Dose/fraction (Gy)'].std(),2))


# In[ ]:




