import matplotlib.pyplot as plt
from pylab import *

class PlotCharts:
    def __init__(self, resultDict):
        self.user =  resultDict['user']
        self.preferences = resultDict['preferences']
    def categoryNcount(self):
        categories =[]
        count = []
        for key in self.preferences:
            categories.append(key)
            count.append(self.preferences[key]['total'])
        # print ("category")
        # print (category)
        # print("count")
        # print(count)
        return categories,count
    
    def getPrecentages(self, category):
        precentage_dict = self.preferences[category]['precentages']
        precent_Label =[]
        precentage =[]
        for key in precentage_dict:
            precent_Label.append(key)
            precentage.append(precentage_dict[key])
        # print("precent_Label")
        # print(precent_Label)
        # print("precentage")
        # print(precentage)
        for label,value in zip(precent_Label,precentage):
            if value == 0:
                precentage.remove(value)
                precent_Label.remove(label)
        return precent_Label, precentage

    def createCharts(self):
        categories,nOf_item = self.categoryNcount()
        grid = len(categories)+1
        subplot(1,grid,1)
        colors = ['gold', 'yellowgreen','skyblue', 'lightcoral']
        pref_plot = plt.pie(nOf_item, labels=categories, autopct='%1.1f%%', shadow=True, startangle=90,colors=colors)        
        title("Preference Areas", fontweight="bold", size =15,y=-0.25)
        for i in range (len(categories)):
            labels,sizes = self.getPrecentages(categories[i])
            plot_title = categories[i]
            subplot(1,grid,i+2)
            plot = plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            title(plot_title, fontweight="bold", size =15,y=-0.25)
        suptitle(self.user, fontweight="bold", fontsize=20)
        show()





