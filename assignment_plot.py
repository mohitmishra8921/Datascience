import matplotlib.pyplot as plt 
colors = ['blue','red','green','yellow']
languages = ["Python","Java ","C++","Ruby"]
usage = [50,40,20,10]
explode =[0,0,0.1,0]
plt.style.use("ggplot")
plt.pie(usage,labels=languages,explode = explode,colors = colors,shadow = True,wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': '-'},autopct='%1.1f%%')
plt.title("Languages Comparison")
plt.show()