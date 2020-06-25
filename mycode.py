import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno 
import numpy as np
import os
import random
from math import pi

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:d}".format(absolute)

def plotter(option):

	arr=os.listdir('upload')

	xcel=pd.ExcelFile('upload/'+arr[-1])
	LIST=xcel.sheet_names
	df=xcel.parse(LIST[-1])
	if len(LIST)==1:
		if LIST[-1]=='Employee Contact Details':
			df.rename(columns={'Current City':'C_City','Current State':'C_State','Permanent City':'P_City',
				'Permanent State':'P_State'}, inplace=True)
			
			df['Tenure in years'] = df['Tenure in years'].apply(np.int64)
			df['Age in years'] = df['Age in years'].apply(np.int64)


			temp = df[df.SBU == 'Bawal']
			xtemp=temp[['C_City', 'C_State']]
			xtemp=pd.concat([xtemp[xtemp['C_City'].notnull()],xtemp[xtemp['C_State'].notnull()]])
			xtemp = xtemp.loc[~xtemp.index.duplicated(keep='first')]

			plt.clf()
			ccred=xtemp['C_City'].value_counts() 
			indexNames = ccred[ ccred <2 ].index
			ccred.drop(indexNames , inplace=True)
			labels =ccred.index.tolist() 
			sizes = ccred.tolist()
			explode=[]
			for i in range(len(sizes)):
			    if random.random()>.7:
			        explode.append(0.1)
			    else:
			        explode.append(0)
			title = plt.title('Regional Diversity in Bawal Plant - Current Address')
			title.set_ha("center")
			plt.gca().axis("equal")
			pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
			plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
			plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
			plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
			plt.tight_layout()
			if 'pdf' in option:
				plt.savefig('download/goo4.pdf', bbox_inches='tight',dpi=200)
			if 'eps' in option:
				plt.savefig('download/goo4.eps', bbox_inches='tight',dpi=200)
			if 'png' in option:
				plt.savefig('download/goo4.png', bbox_inches='tight', dpi=200)

			plt.clf()
			scred=xtemp['C_State'].value_counts() 
			indexNames = scred[ scred <2 ].index
			scred.drop(indexNames , inplace=True)
			labels =scred.index.tolist() 
			sizes = scred.tolist()
			explode=[]
			for i in range(len(sizes)):
			    if random.random()>.7:
			        explode.append(0.1)
			    else:
			        explode.append(0)
			title = plt.title('Regional Diversity in Bawal Plant - Current Address')
			title.set_ha("center")
			plt.gca().axis("equal")
			pie = plt.pie(sizes, explode=explode, autopct=lambda pct: func(pct, sizes),startangle=90,shadow=False)
			plt.legend(pie[0],labels, bbox_to_anchor=(0.85,1.025), loc="upper left")
			plt.gcf().text(0.93,0.04,"* Not available are ignored", ha="right")
			plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
			plt.tight_layout()
			if 'pdf' in option:
				plt.savefig('download/goo3.pdf', bbox_inches='tight',dpi=200)
			if 'eps' in option:
				plt.savefig('download/goo3.eps', bbox_inches='tight',dpi=200)
			if 'png' in option:
				plt.savefig('download/goo3.png', bbox_inches='tight', dpi=200)

			plt.clf()
			ytemp=temp[['P_City', 'P_State']]
			ytemp=pd.concat([ytemp[ytemp['P_City'].notnull()],ytemp[ytemp['P_State'].notnull()]])
			ytemp = ytemp.loc[~ytemp.index.duplicated(keep='first')]
			cpred=ytemp['P_City'].value_counts() 
			indexNames = cpred[ cpred <5 ].index
			cpred.drop(indexNames , inplace=True)
			labels =cpred.index.tolist() 
			sizes = cpred.tolist()
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
			if 'pdf' in option:
				plt.savefig('download/goo2.pdf', bbox_inches='tight',dpi=200)
			if 'eps' in option:
				plt.savefig('download/goo2.eps', bbox_inches='tight',dpi=200)
			if 'png' in option:
				plt.savefig('download/goo2.png', bbox_inches='tight', dpi=200)

			plt.clf()
			spred=ytemp['P_State'].value_counts() 
			indexNames = spred[ spred <4 ].index
			spred.drop(indexNames , inplace=True)
			labels =spred.index.tolist() 
			sizes = spred.tolist()
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
			if 'pdf' in option:
				plt.savefig('download/goo1.pdf', bbox_inches='tight',dpi=200)
			if 'eps' in option:
				plt.savefig('download/goo1.eps', bbox_inches='tight',dpi=200)
			if 'png' in option:
				plt.savefig('download/goo1.png', bbox_inches='tight', dpi=200)

			plt.clf()
			plt.style.use('ggplot')

			ax = df.groupby('Age in years').mean().sort_values("Age in years").plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal')

			# .patches is everything inside of the chart
			for rect in ax.patches:
			    # Find where everything is located
			    height = rect.get_height()
			    width = rect.get_width()
			    x = rect.get_x()
			    y = rect.get_y()

			    # The width of the bar is the data value and can used as the label
			    label_text = f'{height:.0f}'  # f'{height:.2f}' to format decimal values
			    
			    # ax.text(x, y, text)
			    label_x = x + width - 0.2  # adjust 0.2 to center the label
			    label_y = y + height / 2
			    ax.text(label_x, label_y, label_text, ha='right', va='center', fontsize=8)
			    

			ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
			ax.set_ylabel("Tenure in years", fontsize=18)
			ax.set_xlabel("Age bracket", fontsize=18)
			plt.tight_layout()
			if 'pdf' in option:
				plt.savefig('download/goo.pdf', bbox_inches='tight',dpi=200)
			if 'eps' in option:
				plt.savefig('download/goo.eps', bbox_inches='tight',dpi=200)
			if 'png' in option:
				plt.savefig('download/goo.png', bbox_inches='tight', dpi=200)
			return 1
		else:
			listing=['Month','Designation','Department','SBU','Tenure (Years)']
			j=0
			plt.clf()
			for i in listing:
			    temp=df[df.columns[(df.values==i).any(0)]]
			    temp.dropna(inplace=True)
			    temp.columns=temp.iloc[0]
			    temp=temp.drop(temp.index[0])
			    temp=temp[i].value_counts()

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
			    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
			    plt.ylim(0,1.2*max(values))
			     
			    # Plot data
			    ax.plot(angles, values, linewidth=1, linestyle='solid',color='y')
			     
			    # Fill area
			    ax.fill(angles, values, 'b', alpha=0.1)
			    plt.title(i)
			    if 'pdf' in option:
			    	plt.savefig('download/ioo'+str(j)+'.pdf', bbox_inches='tight',dpi=200)
			    if 'eps' in option:
			    	plt.savefig('download/ioo'+str(j)+'.eps', bbox_inches='tight',dpi=200)
			    if 'png' in option:
			    	plt.savefig('download/ioo'+str(j)+'.png', bbox_inches='tight',dpi=200)
			    plt.clf()
			    j=j+1
			return 4

	if len(LIST)==5:
		df=xcel.parse('Main Sheet')
		df.columns = df.iloc[0]
		df.columns.name = None
		ll=df.columns.tolist()
		df = df.loc[:, df.columns.notnull()]
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
		    abhikeliye.index = pd.date_range(start='4/1/2015', periods=len(abhikeliye), freq='M')
		    plt.clf()
		    abhikeliye.plot()
		    plt.ylabel("Productivity per Person per plant", fontsize=11)
		    plt.xlabel("Months", fontsize=11)
		    plt.title(FangShui[i])
		    if 'pdf' in option:
		        plt.savefig('download/hoo'+str(i)+'.pdf', bbox_inches='tight',dpi=200)
		    if 'eps' in option:
		        plt.savefig('download/hoo'+str(i)+'.eps', bbox_inches='tight',dpi=200)
		    if 'png' in option:
		        plt.savefig('download/hoo'+str(i)+'.png', bbox_inches='tight', dpi=200)
		    plt.clf()
		    dcf5=pd.concat([dcf5,abhikeliye], axis=1)
		    
		dcf5.plot()
		plt.legend(labels=FangShui,loc=2, prop={'size': 5})
		plt.ylabel("Productivity per Person per plant", fontsize=11)
		plt.xlabel("Months", fontsize=11)
		option=['png','pdf','eps']
		if 'pdf' in option:
		    plt.savefig('download/hoo.pdf', bbox_inches='tight',dpi=200)
		if 'eps' in option:
		    plt.savefig('download/hoo.eps', bbox_inches='tight',dpi=200)
		if 'png' in option:
		    plt.savefig('download/hoo.png', bbox_inches='tight', dpi=200)
		return 2

	if len(LIST)==10:
		df=xcel.parse('On rolls')
		temp=df[df.columns[(df.values=='Qualification Category').any(0)]]
		temp.dropna(inplace=True)
		temp.columns=temp.iloc[0]
		temp=temp.drop(temp.index[0])
		temp=temp['Qualification Category'].value_counts()

		categories=temp.index.tolist()
		plt.clf()
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
		plt.title('Qualification')
		if 'pdf' in option:
		    plt.savefig('download/joo.pdf', bbox_inches='tight',dpi=200)
		if 'eps' in option:
		    plt.savefig('download/joo.eps', bbox_inches='tight',dpi=200)
		if 'png' in option:
		    plt.savefig('download/joo.png', bbox_inches='tight', dpi=200)
		plt.clf()
		return 5



	first_column="Personnel Cost "+LIST[-1]
	df.rename(columns={first_column:'Expenses'}, inplace=True)

	temp = df[df.Expenses == 'Grand Total']
	ho_variance_percentage=(temp.iloc[0]['Unnamed: 5']/temp.iloc[0]['Unnamed: 3'])*100
	bawal_variance_percentage=(temp.iloc[0]['Unnamed: 11']/temp.iloc[0]['Unnamed: 9'])*100
	chennai_variance_percentage=(temp.iloc[0]['Unnamed: 17']/temp.iloc[0]['Unnamed: 15'])*100
	roorkee_variance_percentage=(temp.iloc[0]['Unnamed: 23']/temp.iloc[0]['Unnamed: 21'])*100
	t16_variance_percentage=(temp.iloc[0]['Unnamed: 29']/temp.iloc[0]['Unnamed: 27'])*100
	npau_variance_percentage=(temp.iloc[0]['Unnamed: 35']/temp.iloc[0]['Unnamed: 33'])*100
	kau_variance_percentage=(temp.iloc[0]['Unnamed: 41']/temp.iloc[0]['Unnamed: 39'])*100
	bau_variance_percentage=(temp.iloc[0]['Unnamed: 47']/temp.iloc[0]['Unnamed: 45'])*100
	gujarat_variance_percentage=(temp.iloc[0]['Unnamed: 53']/temp.iloc[0]['Unnamed: 51'])*100
	total_variance_percentage=(temp.iloc[0]['Unnamed: 59']/temp.iloc[0]['Unnamed: 57'])*100

	budget_lf=[round(temp.iloc[0]['Unnamed: 2'],2),round(temp.iloc[0]['Unnamed: 8'],2),round(temp.iloc[0]['Unnamed: 14'],2),
	round(temp.iloc[0]['Unnamed: 20'],2),round(temp.iloc[0]['Unnamed: 26'],2),round(temp.iloc[0]['Unnamed: 32'],2),
	round(temp.iloc[0]['Unnamed: 38'],2),round(temp.iloc[0]['Unnamed: 44'],2),round(temp.iloc[0]['Unnamed: 50'],2),round(temp.iloc[0]['Unnamed: 56'],2)]

	actual=[round(temp.iloc[0]['Unnamed: 3'],2),round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),
	round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 33'],2),
	round(temp.iloc[0]['Unnamed: 39'],2),round(temp.iloc[0]['Unnamed: 45'],2),round(temp.iloc[0]['Unnamed: 51'],2),round(temp.iloc[0]['Unnamed: 57'],2)]

	variance_lf=[round(temp.iloc[0]['Unnamed: 5'],2),round(temp.iloc[0]['Unnamed: 11'],2),round(temp.iloc[0]['Unnamed: 17'],2),
	round(temp.iloc[0]['Unnamed: 23'],2),round(temp.iloc[0]['Unnamed: 29'],2),round(temp.iloc[0]['Unnamed: 35'],2),
	round(temp.iloc[0]['Unnamed: 41'],2),round(temp.iloc[0]['Unnamed: 47'],2),round(temp.iloc[0]['Unnamed: 53'],2),round(temp.iloc[0]['Unnamed: 59'],2)]

	budget_lf = [int(element // 100000) for element in budget_lf]
	actual = [int(element // 100000) for element in actual]
	variance_lf = [int(element // 100000) for element in variance_lf]

	plant_names = ['HO', 'Bawal', 'Chennai', 'Roorkee', 'T16','NPAU','KAU','BAU','Gujarat','Total']
	variance_PER = [round(ho_variance_percentage,2),round(bawal_variance_percentage,2),round(chennai_variance_percentage,2),round(roorkee_variance_percentage,2),round(t16_variance_percentage,2),round(npau_variance_percentage,2),round(kau_variance_percentage,2),round(bau_variance_percentage,2),round(gujarat_variance_percentage,2),round(total_variance_percentage,2)]
	width=0.20
	x = np.arange(len(plant_names))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, variance_PER,width,color='magenta')
	ax.set_ylabel('Percentage')
	ax.set_title('Percentage Variance LF per plant('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(plant_names)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 9),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	plt.axhline(y=0, color='black', linestyle='-')
	if 'pdf' in option:
		plt.savefig('download/foo.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo.png', bbox_inches='tight', dpi=200)

	width = 0.30  # the width of the bars
	x = np.arange(len(plant_names))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x - 3*width/2, budget_lf, width, label='Budget LF')
	rects2 = ax.bar(x -width/2, actual, width, label='Actual')
	rects3 = ax.bar(x + width/2, variance_lf, width, label='Variance LF')


	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Rupees in lakhs')
	ax.set_title('Budget vs Actual vs Variance('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(plant_names)
	ax.legend(loc='upper center')
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	for rect in rects2:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	for rect in rects3:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo1.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo1.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo1.png', bbox_inches='tight', dpi=200)

	turnover=[88708.9,54311.6,15639.6,16313.1,174973.2]
	label=['Bawal','Chennai','Roorkee','Taloja','Total Turnover FY 1819']

	temp = df[df.Expenses == 'TOTAL (G)']

	cost=[round(temp.iloc[0]['Unnamed: 3'],2),round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 33'],2),round(temp.iloc[0]['Unnamed: 39'],2),round(temp.iloc[0]['Unnamed: 45'],2),round(temp.iloc[0]['Unnamed: 51'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	# cost=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	cost = [int(element // 100000) for element in cost]
	width=0.20
	x = np.arange(len(plant_names))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, cost,width)
	ax.set_ylabel('Rupees in lakhs')
	ax.set_title('Cost of plant ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(plant_names)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo2.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo2.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo2.png', bbox_inches='tight', dpi=200)

	temp = df[df.Expenses == 'Canteen']
	Canteen=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	Canteen=[round((a/b)/1000,2) for a,b in zip(Canteen,turnover)]
	width=0.20

	x = np.arange(len(label))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, Canteen,width)
	ax.set_ylabel('Percentage')
	ax.set_title('Percentage Turnover of Canteen ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(label)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo3.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo3.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo3.png', bbox_inches='tight', dpi=200)


	temp = df[df.Expenses == 'Gifts']
	Gifts=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	Gifts=[round((a/b)/1000,2) for a,b in zip(Gifts,turnover)]

	x = np.arange(len(label))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, Gifts,width)
	ax.set_ylabel('Percentage')
	ax.set_title('Percentage Turnover of Gifts ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(label)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo4.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo4.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo4.png', bbox_inches='tight', dpi=200)


	temp = df[df.Expenses == 'Insurance']
	Insurance=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	Insurance=[round((a/b)/1000,2) for a,b in zip(Insurance,turnover)]

	x = np.arange(len(label))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, Insurance,width)
	ax.set_ylabel('Percentage')
	ax.set_title('Percentage Turnover of Insurance ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(label)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo5.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo5.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo5.png', bbox_inches='tight', dpi=200)


	temp = df[df.Expenses == 'Vehicles For Staff at plant']
	Vehicles=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	Vehicles=[round((a/b)/1000,2) for a,b in zip(Vehicles,turnover)]


	x = np.arange(len(label))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, Vehicles,width)
	ax.set_ylabel('Percentage')
	ax.set_title('Percentage Turnover of Vehicles ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(label)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo6.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo6.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo6.png', bbox_inches='tight', dpi=200)


	temp = df[df.Expenses == 'Bus Hiring For Staff-Plant']
	Bus_Hiring=[round(temp.iloc[0]['Unnamed: 9'],2),round(temp.iloc[0]['Unnamed: 15'],2),round(temp.iloc[0]['Unnamed: 21'],2),round(temp.iloc[0]['Unnamed: 27'],2),round(temp.iloc[0]['Unnamed: 57'],2)]
	Bus_Hiring=[round((a/b)/1000,2) for a,b in zip(Bus_Hiring,turnover)]

	x = np.arange(len(label))
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, Bus_Hiring,width)
	ax.set_ylabel('Rupees in lakhs')
	ax.set_title('Percentage Turnover of Bus ('+LIST[-1]+')')
	ax.set_xticks(x)
	ax.set_xticklabels(label)
	ax.legend()
	for rect in rects1:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
	fig.tight_layout()

	fig.set_size_inches(18.5, 10.5, forward=True)
	if 'pdf' in option:
		plt.savefig('download/foo7.pdf', bbox_inches='tight',dpi=200)
	if 'eps' in option:
		plt.savefig('download/foo7.eps', bbox_inches='tight',dpi=200)
	if 'png' in option:
		plt.savefig('download/foo7.png', bbox_inches='tight', dpi=200)
	return 3