import matplotlib.pyplot as plt
years = [1990,1991,1992,1993,1994]
sachin = [100,200,45,56,87]
plt.bar(years,sachin,color = "green")
plt.xlabel("Years")
plt.ylabel("Runs")
plt.title("Sachin Tendulkar's Yearly Runs")
plt.show()
