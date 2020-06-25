import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno 
import numpy as np

import os
import random
import plotly.express as px
from math import pi

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='center')

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:d}".format(absolute)


# arr=os.listdir('upload')
# x=pd.ExcelFile('upload/'+arr[-1])

xcel=pd.ExcelFile("Auto Manpower Details Apr'20 (Updated).xlsx")

print(xcel.sheet_names)

#temp=x.parse(LIST[-1])
df=xcel.parse('On rolls')

temp=df[df.columns[(df.values=='Qualification Category').any(0)]]
temp.dropna(inplace=True)
temp.columns=temp.iloc[0]
temp=temp.drop(temp.index[0])
temp=temp['Qualification Category'].value_counts()

categories=temp.index.tolist()

N=len(categories)

values=temp.values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)

plt.ylim(0,1.2*max(values))
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid',color='y')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

plt.show()























# df.columns = df.iloc[0]
# df.columns.name = None
# ll=df.columns.tolist()
# df = df.loc[:, df.columns.notnull()]

# first_column="Personnel Cost "+LIST[-2]
# temp.rename(columns={first_column:'Expenses'}, inplace=True)
# df.rename(columns={'Current City':'C_City','Current State':'C_State','Permanent City':'P_City',
#     'Permanent State':'P_State'}, inplace=True)
# temp = temp[temp.Expenses == 'Grand Total']
df = df.drop(['General','FY 09-10','FY 10-11','FY 11-12','FY 12-13','FY 13-14',
    'FY 14-15','FY 15-16'], axis=1)
df = df.drop([245,0.06857314158451536,0.11217833021197235,0,0,0,97.21,97.21,97.21,0,
    0,0,311.9978368883191,311.9978368883191,311.9978368883191,1227.6235106649574],
     axis=1)
df.rename(columns={'Bawal':'Fields'}, inplace=True)
df.index=df.iloc[:,0]
df = df.loc[df.index.notnull(),:]
df.index.name = None
df = df.drop(['Fields','FY 15~16','FY 16~17','FY 1718','FY 1819',"YTD Mar'19"],
    axis=1)

FangShui=['Bawal','Chennai','Roorkee','T-16 Auto','Auto']
dcf5=pd.DataFrame()
for i in range(5):
    if i==4:
        abhikeliye=df.loc[FangShui[i]:,:]
    else:
        abhikeliye=df.loc[FangShui[i]:FangShui[i+1],:]
    abhikeliye=abhikeliye.iloc[7,:]
    abhikeliye=abhikeliye.iloc[:48]
    abhikeliye=abhikeliye.astype(str).astype(float).apply(np.int64)
    abhikeliye.index = pd.date_range(start='4/1/2015', periods=len(data), freq='M')
    abhikeliye.plot()
    plt.ylabel("Productivity per Person per plant", fontsize=18)
    plt.xlabel("Months", fontsize=18)
    plt.tight_layout()
    plt.set_size_inches(18.5, 10.5, forward=True)
    if 'pdf' in option:
        plt.savefig('download/foo4.pdf', bbox_inches='tight',dpi=200)
    if 'eps' in option:
        plt.savefig('download/foo4.eps', bbox_inches='tight',dpi=200)
    if 'png' in option:
        plt.savefig('download/foo4.png', bbox_inches='tight', dpi=200)
    dcf5=pd.concat([dcf5,abhikeliye], axis=1)
    
dcf5.plot()
plt.legend(labels=FangShui,loc=2, prop={'size': 5})
plt.ylabel("Productivity per Person per plant", fontsize=11)
plt.xlabel("Months", fontsize=11)
option=['png','pdf','eps']
if 'pdf' in option:
    plt.savefig('download/hoo4.pdf', bbox_inches='tight',dpi=200)
if 'eps' in option:
    plt.savefig('download/hoo4.eps', bbox_inches='tight',dpi=200)
if 'png' in option:
    plt.savefig('download/hoo4.png', bbox_inches='tight', dpi=200)

# temp = df[df.SBU == 'Bawal']
# msno.matrix(temp)
# plt.show()

# temp=pd.concat([temp[temp['C_City'].notnull()],
#     temp[temp['P_City'].notnull()],
#     temp[temp['C_State'].notnull()],
#     temp[temp['P_State'].notnull()]]).drop_duplicates().reset_index(drop=True)

# xtemp=temp[['C_City', 'C_State']]
# # msno.matrix(xtemp)
# # plt.show()
# xtemp=pd.concat([xtemp[xtemp['C_City'].notnull()],xtemp[xtemp['C_State'].notnull()]])
# xtemp = xtemp.loc[~xtemp.index.duplicated(keep='first')]

# cred=xtemp['C_City'].value_counts() 
# # Get names of indexes for which column Age has value 30
# indexNames = cred[ cred <2 ].index
 
# # Delete these row indexes from dataFrame
# cred.drop(indexNames , inplace=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels =cred.index.tolist() 
# sizes = cred.tolist()
# explode=[]
# for i in range(len(sizes)):
#     if random.random()>.7:
#         explode.append(0.1)
#     else:
#         explode.append(0)

# title = plt.title('Regional Diversity in Bawal Plant - Current Address')
# title.set_ha("center")
# plt.gca().axis("equal")
# pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
# plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
# plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
# plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
# plt.tight_layout()
# plt.show()

# cred=xtemp['C_State'].value_counts() 
# # Get names of indexes for which column Age has value 30
# indexNames = cred[ cred <2 ].index
 
# # Delete these row indexes from dataFrame
# cred.drop(indexNames , inplace=True)

# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels =cred.index.tolist() 
# sizes = cred.tolist()
# explode=[]
# for i in range(len(sizes)):
#     if random.random()>.7:
#         explode.append(0.1)
#     else:
#         explode.append(0)

# title = plt.title('Regional Diversity in Bawal Plant - Current Address')
# title.set_ha("center")
# plt.gca().axis("equal")
# pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
# plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
# plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
# plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
# plt.tight_layout()
# plt.show()




# xtemp=temp[['P_City', 'P_State']]
# # msno.matrix(xtemp)
# # plt.show()
# xtemp=pd.concat([xtemp[xtemp['P_City'].notnull()],xtemp[xtemp['P_State'].notnull()]])
# xtemp = xtemp.loc[~xtemp.index.duplicated(keep='first')]
# cred=xtemp['P_City'].value_counts() 
# # Get names of indexes for which column Age has value 30
# indexNames = cred[ cred <5 ].index
 
# # Delete these row indexes from dataFrame
# cred.drop(indexNames , inplace=True)

# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels =cred.index.tolist() 
# sizes = cred.tolist()
# explode=[]
# for i in range(len(sizes)):
#     if random.random()>.7:
#         explode.append(0.1)
#     else:
#         explode.append(0)

# title = plt.title('Regional Diversity in Bawal Plant - Permanent Address')
# title.set_ha("center")
# plt.gca().axis("equal")
# pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
# plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
# plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
# plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
# plt.tight_layout()
# plt.show()



# cred=xtemp['P_State'].value_counts() 
# # Get names of indexes for which column Age has value 30
# indexNames = cred[ cred <4 ].index
 
# # Delete these row indexes from dataFrame
# cred.drop(indexNames , inplace=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels =cred.index.tolist() 
sizes = cred.tolist()
explode=[]
for i in range(len(sizes)):
    if random.random()>.7:
        explode.append(0.1)
    else:
        explode.append(0)

title = plt.title('Regional Diversity in Bawal Plant - Permanent Address')
title.set_ha("center")
plt.gca().axis("equal")
pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
plt.tight_layout()
plt.show()







# fig1, ax1 = plt.subplots()
# def func(pct, allvals):
#     absolute = int(pct/100.*np.sum(allvals))
#     return "{:d}".format(absolute)

# ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda pct: func(pct, sizes),
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.tight_layout()
# plt.show()

# import matplotlib.patches

# plt.title('Regional Diversity in Bawal Plant - Curr')
# plt.gca().axis("equal")
# pie = plt.pie(sizes, startangle=90,
#                             wedgeprops = { 'linewidth': 2, "edgecolor" :"k" })
# handles = []
# for i, l in enumerate(labels):
#     handles.append(matplotlib.patches.Patch(color=plt.cm.Set3((i)/8.), label=l))
# plt.legend(handles,labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
# plt.gcf().text(0.93,0.04,"* out of competition since 2006", ha="right")
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
# plt.show()


# cred.plot.pie(label=, title="Daily visitors to the village library");


# cred = xtemp.groupby('C_City')['C_State'].nunique()

# x_temp=xtemp.set_index('C_State')

# plt.style.use('ggplot')

# ax = df.plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal')

# # .patches is everything inside of the chart
# for rect in ax.patches:
#     # Find where everything is located
#     height = rect.get_height()
#     width = rect.get_width()
#     x = rect.get_x()
#     y = rect.get_y()

#     # The width of the bar is the data value and can used as the label
#     label_text = f'{height:.0f}'  # f'{height:.2f}' to format decimal values
    
#     # ax.text(x, y, text)
#     label_x = x + width - 0.2  # adjust 0.2 to center the label
#     label_y = y + height / 2
#     ax.text(label_x, label_y, label_text, ha='right', va='center', fontsize=8)
    

# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)    
# ax.set_ylabel("Tenure in years", fontsize=18)
# ax.set_xlabel("Age bracket", fontsize=18)
# plt.show()




xtemp=temp[['P_State', 'P_City']]
xtemp=pd.concat([xtemp[xtemp['P_State'].notnull()],xtemp[xtemp['P_City'].notnull()]])
xtemp = xtemp.loc[~xtemp.index.duplicated(keep='first')]

# budget_lf=[round(temp.iloc[0]['Unnamed: 2'],2),round(temp.iloc[0]['Unnamed: 8'],2),round(temp.iloc[0]['Unnamed: 14'],2),
# round(temp.iloc[0]['Unnamed: 20'],2),round(temp.iloc[0]['Unnamed: 26'],2),round(temp.iloc[0]['Unnamed: 32'],2),
# round(temp.iloc[0]['Unnamed: 38'],2),round(temp.iloc[0]['Unnamed: 44'],2),round(temp.iloc[0]['Unnamed: 50'],2),round(temp.iloc[0]['Unnamed: 56'],2)]

# actual=[round(temp.iloc[0]['Unnamed: 3'],2),round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),
# round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 33'],2),
# round(temp.iloc[0]['Unnamed: 39'],2),round(temp.iloc[0]['Unnamed: 45'],2),round(temp.iloc[0]['Unnamed: 51'],2),round(temp.iloc[0]['Unnamed: 57'],2)]

# variance_lf=[round(temp.iloc[0]['Unnamed: 5'],2),round(temp.iloc[0]['Unnamed: 11'],2),round(temp.iloc[0]['Unnamed: 17'],2),
# round(temp.iloc[0]['Unnamed: 23'],2),round(temp.iloc[0]['Unnamed: 29'],2),round(temp.iloc[0]['Unnamed: 35'],2),
# round(temp.iloc[0]['Unnamed: 41'],2),round(temp.iloc[0]['Unnamed: 47'],2),round(temp.iloc[0]['Unnamed: 53'],2),round(temp.iloc[0]['Unnamed: 59'],2)]

# budget_lf = [int(element // 100000) for element in budget_lf]
# actual = [int(element // 100000) for element in actual]
# variance_lf = [int(element // 100000) for element in variance_lf]


# plant_names = ['HO', 'Bawal', 'Chennai', 'Roorkee', 'T16','NPAU','KAU','BAU','Gujarat','Total']


# df=x.parse(LIST[-1])
# first_column="Personnel Cost "+LIST[-1]
# df.rename(columns={first_column:'Expenses'}, inplace=True)

# turnover=[88708.9,54311.6,15639.6,16313.1,174973.2]
# label=['Bawal','Chennai','Roorkee','Taloja','Total Turnover FY 1819']

# temp = df[df.Expenses == 'Canteen']
# Canteen=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
# Canteen=[round((a/b)/1000,2) for a,b in zip(Canteen,turnover)]

# temp = df[df.Expenses == 'Gifts']
# Gifts=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
# Gifts=[round((a/b)/1000,2) for a,b in zip(Gifts,turnover)]

# temp = df[df.Expenses == 'Insurance']
# Insurance=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
# Insurance=[round((a/b)/1000,2) for a,b in zip(Insurance,turnover)]

# temp = df[df.Expenses == 'Vehicles For Staff at plant']
# Vehicles=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
# Vehicles=[round((a/b)/1000,2) for a,b in zip(Vehicles,turnover)]

# temp = df[df.Expenses == 'Bus Hiring For Staff-Plant']
# Bus_Hiring=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
# Bus_Hiring=[round((a/b)/1000,2) for a,b in zip(Bus_Hiring,turnover)]














# x = np.arange(len(plant_names))  # the label locations
# width = 0.20  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 30*width/20, budget_lf, width, label='Budget LF')
# rects2 = ax.bar(x -10*width/20, actual, width, label='Actual')
# rects3 = ax.bar(x + 10*width/20, variance_lf, width, label='Variance LF')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Rupees')
# ax.set_title('Budget vs Actual vs Variance')
# ax.set_xticks(x)
# ax.set_xticklabels(plant_names)
# ax.legend()

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# fig.tight_layout()

# plt.show()


# fig.set_size_inches(18.5, 10.5, forward=True)
# if 'pdf' in option:
# 	plt.savefig('download/foo.pdf', bbox_inches='tight',dpi=200)
# if 'eps' in option:
# 	plt.savefig('download/foo.eps', bbox_inches='tight',dpi=200)
# if 'png' in option:
# 	plt.savefig('download/foo.png', bbox_inches='tight', dpi=200)

# #plt.show()

# # del temp['_']
# # del temp['_.1']
# # del temp['_.2']
# # del temp['_.3']
# # del temp['_.4']
# # del temp['_.5']
# # del temp['_.6']
# # del temp['_.7']
# # del temp['_.8']


# # temp = temp[temp.Expenses.notnull()]
# # temp.at[28,'Expenses']='TOTAL SubContract( E)'
# # temp.at[33,'Expenses']='TOTAL Recruitment(F)'
# # temp.at[52,'Expenses']='TOTAL Welfare(G)'

# # msno.matrix(temp)
# # plt.show()

# # temp = temp[temp.Expenses != 'Recruitment & Training']
# # temp = temp[temp.Expenses != 'Sub Contractors, Leave Encashment, Bonus, PLI']
# # temp = temp[temp.Expenses != 'Warehouses']
# # applying get_value() function 



# # temp=temp.assign(Month="YTD Mar'19")
# # temp=temp.fillna(0)
# # temp=temp.assign(Month="YTD Mar'19")

# # df=df.append(temp)
# # Gtotal=df.append(temp)
# # Gtotal.to_csv('GTotal',index=False)


# # files = glob.glob('/home/ajitkumar/Documents/code/python/Flask/AIS/upload/*')
# # for f in files:
# #     os.remove(f)

A = [45, 17, 47]
B = [91, 70, 72]
C = [68, 43, 13]

# pandas dataframe
df = pd.DataFrame(data={'A': A, 'B': B, 'C': C})
df.index = ['C1', 'C2', 'C3']

     A   B   C
C1  45  91  68
C2  17  70  43
C3  47  72  13
















