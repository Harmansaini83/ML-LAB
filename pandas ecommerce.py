import pandas as pd   # importing pandas library for data analysis


df = pd.read_csv('Ecommerce Purchases.csv')   # Q1: loading the dataset into a DataFrame


print("rows and columns:", df.shape)   # Q2: .shape returns (number of rows, number of columns)


average = df['Purchase Price'].mean()   # Q3: calculates the average of the Purchase Price column
print("Average Purchase Price:", average)


highest = df['Purchase Price'].max()    # Q4: finds the highest purchase price
lowest = df['Purchase Price'].min()     # Q4: finds the lowest purchase price
print("Highest Price:", highest)
print("Lowest Price:", lowest)


lawyer_count = df[df['Job'] == 'Lawyer'].shape[0]   # Q5: filters rows where Job='Lawyer' and counts them
print("People with the job title 'Lawyer':", lawyer_count)


am_pm_counts = df['AM or PM'].value_counts()   # Q6: counts how many purchases were made in AM vs PM
print("AM/PM purchase counts:")
print(am_pm_counts)


top_5_jobs = df['Job'].value_counts().head(5)   # Q7: finds the 5 most common job titles using value_counts()
print("Top 5 most common job titles:")
print(top_5_jobs)


email = df[df['Credit Card'] == 4926535242672853]['Email'].iloc[0]  
# Q8: filters the row with the given credit card number, selects the 'Email' column, takes the first match
print(f"Email for Credit Card 4926535242672853: {email}")


lotprice = df[df['Lot'] == '90 WT']['Purchase Price'].iloc[0]  
# Q9: filters the row where 'Lot' = '90 WT', selects purchase price, extracts the first match
print(f"Purchase Price for Lot '90 WT': {lotprice}")


america = df[(df['CC Provider'] == 'American Express') & (df['Purchase Price'] > 95)].shape[0]  
# Q10: filters rows where CC Provider is American Express AND Purchase Price > 95, counts how many
print(f"American Express users with purchase > $95: {america}")
