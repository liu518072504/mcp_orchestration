import pandas as pd

# Read the CSV file using the full path
df = pd.read_csv(r'ragDatabase\excel\Expedited_Orders.csv')

# Count high priority expedited orders using 'priority_level' column
high_priority_count = len(df[df['priority_level'] == 'High'])

# Print the result
print(f'Number of high priority expedited orders: {high_priority_count}')