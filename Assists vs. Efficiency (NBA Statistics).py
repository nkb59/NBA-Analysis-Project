#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_csv(r"/Users/niketanbaranwal/Downloads/NBA_Team_Stats.csv")


# In[66]:


df


# In[4]:


#Indexing by Year 
df2 = df.set_index('Year')
df2


# In[5]:


#Filtering the data as we want to compare average assists per team to efficency 
df2.filter(items = ['Team','Ast', 'Eff'], axis = 1)


# In[6]:


df3 = df.set_index('Year')
df3.sort_values(by = 'Ast', ascending = False)
#2016-2017 Warriors averaged the most assist since the 1997-1998 season


# In[7]:


df4 = df.set_index('Year')
df4.sort_values(by = 'Eff', ascending = False)
#2016-2017 Warriors were the most efficient team since the 1997-1998 season


# In[8]:


df5 = df.set_index('Year')
df5.sort_values(by = 'Ast', ascending = True)
#1998-1999 Atlanta Hawks averaged the least assists since the 1997-1998 NBA season


# In[9]:


df6 = df.set_index('Year')
df6.sort_values(by = 'Eff', ascending = True)
#1998-1999 Chicago Bulls were the least efficient since the 1997-1998 season
#1998-1999 Atlanta Hawks were the third least efficient since the 1997-1998 season


# In[10]:


#Average Assists
ast_mean = df[["Ast"]].mean()
ast_mean


# In[11]:


#Average Efficiency
eff_mean = df[["Eff"]].mean()
eff_mean


# In[12]:


#Convert the Assist column into a series 
ser_ast = df['Ast'].squeeze()
ser_ast = df['Ast']
print(ser_ast)


# In[13]:


print(type(ser_ast))


# In[14]:


#Convert the Efficiency column into a series 
ser_eff = df['Eff'].squeeze()
ser_eff = df['Eff']
print(ser_eff)


# In[15]:


print(type(ser_eff))


# In[29]:


#Pearson's R
#r-value > .7 meaning there is a good positive correlation between assist and efficiency
r = pearsonr(ser_ast,ser_eff)


# In[33]:


round(r,2)


# In[37]:


#Spearman's rho
corr = spearmanr(ser_ast,ser_eff)


# In[38]:


corr


# In[40]:


#Kendall's Tau
p = kendalltau(ser_ast,ser_eff)


# In[41]:


p


# In[20]:


#Create a scatterplot for Ast vs. Eff to see correlation
#Looks like a positive correlation without line of best fit 
ax = sns.scatterplot(x="Ast", y="Eff", data= df);
ax.set_title("Assists vs. Efficiency")
ax.set_xlabel("Assists")
ax.set_ylabel("Efficiency")


# In[21]:


#Create line of best fit 
#Positive Correlation as the slope is positive 
sns.lmplot(x="Ast", y="Eff", data=df)


# In[22]:


#Customize plot so that you can differentiate line from data point 
sns.lmplot(x="Ast", y="Eff", data=df, scatter_kws={"color": "black"}, line_kws={"color": "red"});


# In[49]:


data = df[['Ast', 'Eff']]


# In[50]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Ast'], y=data['Eff'])
plt.title('Assists vs Player Efficiency Rating')
plt.xlabel('Assists')
plt.ylabel('Team Efficiency Rating')
plt.show()


# In[51]:


X = data[['Ast']]  # Independent variable
y = data['Eff']  # Dependent variable


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[54]:


# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[55]:


# Predict the dependent variable for the test set
y_pred = model.predict(X_test)


# In[56]:


# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared (R²) to understand the proportion of variance explained by the model
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


# In[60]:


#Visualize regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Regression: Assists vs Player Efficiency Rating')
plt.xlabel('Assists')
plt.ylabel('Player Efficiency Rating')
plt.legend()
plt.show()


# In[58]:


# View the model's coefficients (slope) and intercept
print(f'Coefficient (slope): {model.coef_}')
print(f'Intercept: {model.intercept_}')

