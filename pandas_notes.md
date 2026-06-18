## **Pandas** 

## **Python Data Analysis Library — Complete Notes** 

Theory + Examples + Code | Sahil Pandey | DTU 

## **SECTION 1: Introduction to Pandas** 

## **What is Pandas?** 

Pandas is the most popular Python library for data manipulation and analysis. It provides two primary data structures: Series (1D) and DataFrame (2D), which are built on top of NumPy. Pandas is essential for data cleaning, transformation, exploration, and preparation before feeding data into ML models. 

## **🔑 Why Pandas?** 

Real-world data is messy, tabular, and mixed-type. Pandas handles all of this: missing values, mixed dtypes, reading files (CSV, Excel, JSON, SQL), grouping, merging, and much more — all with intuitive syntax. 

## **Installation & Import** 

```
pip install pandas
import pandas as pd
import numpy as np   # often used together
pd.__version__  # e.g., '2.1.0'
```

## **SECTION 2: Series** 

## **Pandas Series** 

A Series is a one-dimensional labeled array. Think of it as a column in a spreadsheet, or a NumPy array with labels (called an index). 

## **Creating a Series** 

**==> picture [468 x 467] intentionally omitted <==**

**----- Start of picture text -----**<br>
import pandas as pd<br>import numpy as np<br># From a Python list<br>s = pd.Series([10, 20, 30, 40, 50])<br>print(s)<br># 0    10<br># 1    20<br># 2    30<br># 3    40<br># 4    50<br># dtype: int64<br># With custom index<br>s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])<br>print(s)<br># a    10<br># b    20<br># c    30<br># From a dictionary<br>data = {'Math': 95, 'Physics': 88, 'Chemistry': 91}<br>s = pd.Series(data)<br># Math         95<br># Physics       88<br># Chemistry    91<br># From scalar (broadcasts)<br>pd.Series(7, index=[0,1,2,3])   # 0->7, 1->7, 2->7, 3->7<br># From NumPy array<br>pd.Series(np.arange(5))<br>**----- End of picture text -----**<br>


## **Accessing Series Elements** 

```
s = pd.Series([10, 20, 30, 40], index=['a','b','c','d'])
```

**==> picture [468 x 178] intentionally omitted <==**

**----- Start of picture text -----**<br>
# By label (index name)<br>s['a']          # 10<br>s[['a', 'c']]   # a->10, c->30<br># By position<br>s.iloc[0]       # 10 — first element<br>s.iloc[-1]      # 40 — last element<br>s.iloc[1:3]     # b->20, c->30<br># Boolean indexing<br>s[s > 20]       # c->30, d->40<br>**----- End of picture text -----**<br>


## **Series Attributes & Methods** 

**==> picture [468 x 363] intentionally omitted <==**

**----- Start of picture text -----**<br>
s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])<br>s.values       # numpy array [3 1 4 1 5 9 2 6]<br>s.index        # RangeIndex(start=0, stop=8, step=1)<br>s.dtype        # dtype('int64')<br>s.shape        # (8,)<br>s.size         # 8<br>s.name         # None (can set it)<br>s.sum()        # 31<br>s.mean()       # 3.875<br>s.min()        # 1<br>s.max()        # 9<br>s.std()        # standard deviation<br>s.describe()   # count, mean, std, min, 25%, 50%, 75%, max<br>s.sort_values()          # sort by value<br>s.sort_index()           # sort by index<br>s.value_counts()         # frequency count of each value<br>s.unique()               # unique values<br>s.nunique()              # number of unique values<br>s.isnull()               # True where NaN<br>s.dropna()               # remove NaN values<br>s.fillna(0)              # replace NaN with 0<br>s.apply(lambda x: x**2) # apply function to each element<br>**----- End of picture text -----**<br>


## **Series Operations** 

**==> picture [468 x 233] intentionally omitted <==**

**----- Start of picture text -----**<br>
s1 = pd.Series([10, 20, 30], index=['a','b','c'])<br>s2 = pd.Series([1, 2, 3], index=['b','c','d'])<br># Arithmetic — aligns by index, NaN where index doesn't match<br>s1 + s2<br># a     NaN  (a only in s1)<br># b    22.0  (20+2)<br># c    32.0  (30+3)<br># d     NaN  (d only in s2)<br># Use fill_value to handle missing<br>s1.add(s2, fill_value=0)<br># a    10.0<br># b    22.0<br># c    32.0<br># d     3.0<br>**----- End of picture text -----**<br>


## **SECTION 3: DataFrame** 

## **Pandas DataFrame** 

A DataFrame is a 2D labeled data structure — like a spreadsheet or SQL table. It has rows (index) and columns (column labels). Each column is a Series sharing the same index. 

## **Creating a DataFrame** 

**==> picture [468 x 188] intentionally omitted <==**

**----- Start of picture text -----**<br>
# From dictionary of lists (most common)<br>data = {<br>    'Name':   ['Alice', 'Bob', 'Charlie', 'Diana'],<br>    'Age':    [25, 30, 35, 28],<br>    'Score':  [88.5, 92.0, 78.5, 95.0],<br>    'City':   ['Delhi', 'Mumbai', 'Pune', 'Bangalore']<br>}<br>df = pd.DataFrame(data)<br># From list of dictionaries<br>records = [<br>    {'Name': 'Alice', 'Age': 25},<br>    {'Name': 'Bob', 'Age': 30},<br>**----- End of picture text -----**<br>


```
]
df2 = pd.DataFrame(records)
# From NumPy array
arr = np.random.randint(0, 100, (4, 3))
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
# With custom index
df4 = pd.DataFrame(data, index=['r1','r2','r3','r4'])
# From CSV file
df5 = pd.read_csv('data.csv')
```

## **Exploring a DataFrame** 

```
df.head()        # first 5 rows
df.head(10)      # first 10 rows
df.tail()        # last 5 rows
df.tail(3)       # last 3 rows
df.shape         # (4, 4) — (rows, cols)
df.ndim          # 2
df.size          # 16 — total cells
df.dtypes        # data type of each column
df.columns       # Index(['Name','Age','Score','City'])
df.index         # RangeIndex(start=0, stop=4)
df.info()        # summary: dtypes, non-null counts, memory
df.describe()    # stats for numeric columns only
df.describe(include='all')  # stats for all columns
df.memory_usage() # memory usage per column
df.count()        # non-null count per column
```

## **SECTION 4: Selecting Data** 

## **Selecting Columns and Rows** 

## **Selecting Columns** 

```
# Single column — returns Series
df['Name']
```

```
df.Name          # dot notation (works if no spaces)
# Multiple columns — returns DataFrame
df[['Name', 'Score']]
df[['Age', 'City', 'Name']]  # reorder too
```

## **loc — Label-based selection** 

loc selects by row labels (index values) and column names. 

```
# Select single row by label
df.loc[0]           # row with index 0
df.loc[2]           # row with index 2
# Select multiple rows
df.loc[[0, 2]]      # rows 0 and 2
df.loc[0:2]         # rows 0, 1, 2 (INCLUSIVE both ends!)
# Select rows and columns
df.loc[0, 'Name']               # Alice
df.loc[0:2, 'Name']             # Names of rows 0,1,2
df.loc[0:2, ['Name', 'Score']]  # rows 0-2, 2 columns
df.loc[:, 'Age']                # all rows, Age column
# Condition-based (boolean)
df.loc[df['Age'] > 28]
df.loc[df['Score'] > 90, ['Name', 'Score']]
```

## **iloc — Position-based selection** 

iloc selects by integer position (like NumPy indexing). 

```
df.iloc[0]          # first row
```

```
df.iloc[-1]         # last row
df.iloc[0:3]        # rows 0, 1, 2 (exclusive end!)
df.iloc[0, 1]       # row 0, col 1 (Age of Alice)
df.iloc[1:3, 0:2]   # rows 1-2, cols 0-1
df.iloc[:, 2]       # all rows, column 2
df.iloc[[0,2], [1,3]] # rows 0,2 and cols 1,3
```

```
# Select every other row
df.iloc[::2]
```

## **🔑 loc vs iloc** 

loc uses LABELS — row index names and column names. Slicing is INCLUSIVE on both ends. iloc uses POSITIONS — integer indices (0-based). Slicing is EXCLUSIVE on the end (like Python lists). 

## **Boolean / Conditional Filtering** 

```
# Single condition
df[df['Age'] > 28]
df[df['City'] == 'Delhi']
df[df['Score'] >= 90]
# Multiple conditions — use & and | (NOT 'and'/'or')
df[(df['Age'] > 25) & (df['Score'] > 85)]
df[(df['City'] == 'Delhi') | (df['City'] == 'Mumbai')]
# NOT condition
df[~(df['City'] == 'Pune')]
# isin — check if value is in a list
df[df['City'].isin(['Delhi', 'Pune'])]
# between — range filter
df[df['Age'].between(25, 30)]
# String conditions
df[df['Name'].str.startswith('A')]
df[df['Name'].str.contains('li')]
```

## **query() — SQL-like filtering** 

```
# Cleaner syntax for filtering
df.query('Age > 28')
df.query('Age > 25 and Score > 85')
df.query('City == "Delhi" or City == "Mumbai"')
# Reference a variable using @
min_age = 28
df.query('Age > @min_age')
```

## **SECTION 5: Modifying DataFrames** 

## **Adding, Updating & Deleting** 

## **Adding New Columns** 

```
# Add a new column
df['Bonus'] = df['Score'] * 0.1
df['Grade'] = ['A', 'A', 'B', 'A']
# Derived column
df['Score_squared'] = df['Score'] ** 2
# Using assign (returns new df, doesn't modify original)
df2 = df.assign(Tax = df['Score'] * 0.05)
# Using np.where for conditional column
df['Pass'] = np.where(df['Score'] >= 85, 'Pass', 'Fail')
```

## **Updating Values** 

**==> picture [468 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
# Update entire column<br>df['Age'] = df['Age'] + 1<br># Update specific cell by label<br>df.loc[0, 'Score'] = 99.0<br># Update specific cell by position<br>df.iloc[1, 2] = 88.0<br># Update based on condition<br>df.loc[df['Score'] < 80, 'Grade'] = 'C'<br># Replace specific values<br>df['City'].replace('Delhi', 'New Delhi', inplace=True)<br>df.replace({'Mumbai': 'Bombay'})  # multiple<br>**----- End of picture text -----**<br>


```
# Rename columns
```

```
df.rename(columns={'Name': 'Full_Name', 'Score': 'Marks'}, inplace=True)
```

## **Dropping Rows and Columns** 

```
# Drop column
df.drop('Bonus', axis=1, inplace=True)
df.drop(columns=['Bonus', 'Tax'])   # multiple columns
# Drop rows by index label
df.drop(0, axis=0)            # drop row with index 0
df.drop([0, 2])               # drop rows 0 and 2
# Drop rows by condition
df = df[df['Age'] >= 25]  # keep rows where Age >= 25
# Reset index after dropping
df.reset_index(drop=True, inplace=True)
```

## **SECTION 6: Handling Missing Values** 

## **Missing Data (NaN)** 

Real-world datasets almost always have missing values. Pandas represents missing numeric data as NaN (Not a Number) and missing object data as None. 

```
import numpy as np
data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)
# Detect missing values
df.isnull()           # boolean DataFrame — True where NaN
df.isna()             # same as isnull()
df.isnull().sum()     # count of NaN per column
df.isnull().sum().sum() # total NaN count
df.isnull().any()     # True for cols that have any NaN
df.notnull()          # opposite of isnull()
```

```
# Drop rows/columns with NaN
df.dropna()                   # drop rows with ANY NaN
df.dropna(axis=1)             # drop COLUMNS with any NaN
df.dropna(how='all')          # drop only if ALL values are NaN
df.dropna(thresh=2)           # keep rows with at least 2 non-NaN
df.dropna(subset=['A', 'B'])  # drop rows where A or B is NaN
# Fill missing values
df.fillna(0)                  # fill all NaN with 0
df['A'].fillna(df['A'].mean()) # fill with column mean
df.fillna(method='ffill')     # forward fill (use prev value)
df.fillna(method='bfill')     # backward fill (use next value)
df.fillna({'A': 0, 'B': 99})  # different fill per column
# Interpolate — estimate NaN values
df['A'].interpolate()         # linear interpolation
```

## **SECTION 7: Data Types & String Operations** 

## **Working with Data Types** 

```
df.dtypes                   # check all dtypes
# Convert dtypes
df['Age'] = df['Age'].astype(float)
df['Score'] = df['Score'].astype(int)
df['Name'] = df['Name'].astype(str)
# Convert to numeric (handle errors gracefully)
pd.to_numeric(df['col'], errors='coerce')   # NaN on failure
pd.to_numeric(df['col'], errors='ignore')   # keep original on fail
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# Categorical — save memory for low-cardinality columns
df['City'] = df['City'].astype('category')
```

## **String Operations (.str accessor)** 

The .str accessor lets you apply string methods to an entire Series of strings. 

**==> picture [468 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
s = pd.Series(['Alice Smith', 'BOB JONES', '  charlie  ', 'diana123'])<br>s.str.upper()           # uppercase<br>s.str.lower()           # lowercase<br>s.str.strip()           # remove leading/trailing spaces<br>s.str.len()             # length of each string<br>s.str.replace('a','@')  # replace character<br>s.str.contains('li')    # boolean — True if contains 'li'<br>s.str.startswith('A')   # starts with check<br>s.str.endswith('h')     # ends with check<br>s.str.split(' ')        # split by space — returns list<br>s.str.split(' ', expand=True)  # split into multiple columns<br>s.str.extract(r'(\d+)')  # regex extract — gets '123' from 'diana123'<br>s.str.findall(r'\d+')   # find all regex matches<br>**----- End of picture text -----**<br>


## **SECTION 8: Sorting, Ranking & Iteration** 

## **Sorting** 

**==> picture [468 x 207] intentionally omitted <==**

**----- Start of picture text -----**<br>
# Sort by column value<br>df.sort_values('Score')                    # ascending (default)<br>df.sort_values('Score', ascending=False)   # descending<br># Sort by multiple columns<br>df.sort_values(['City', 'Score'], ascending=[True, False])<br># Sort by index<br>df.sort_index()                # ascending index<br>df.sort_index(ascending=False) # descending index<br># Rank — assign ranks<br>df['Score'].rank()             # rank (1 = smallest)<br>df['Score'].rank(ascending=False)  # rank (1 = largest)<br>**----- End of picture text -----**<br>


## **Iteration** 

**==> picture [468 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
# Iterate over columns<br>for col in df.columns:<br>    print(col, df[col].dtype)<br># Iterate over rows (SLOW — avoid for large datasets)<br>for index, row in df.iterrows():<br>    print(index, row['Name'], row['Score'])<br># Faster iteration with itertuples()<br>for row in df.itertuples():<br>    print(row.Name, row.Score)<br>**----- End of picture text -----**<br>


## **SECTION 9: GroupBy** 

## **GroupBy — Split-Apply-Combine** 

GroupBy is one of the most powerful Pandas features. It splits data into groups, applies a function to each group, and combines results. 

**==> picture [468 x 309] intentionally omitted <==**

**----- Start of picture text -----**<br>
data = {<br>    'Name':  ['Alice','Bob','Alice','Bob','Charlie'],<br>    'Month': ['Jan','Jan','Feb','Feb','Jan'],<br>    'Sales': [100, 150, 200, 130, 90],<br>    'Cost':  [40, 60, 70, 50, 30]<br>}<br>df = pd.DataFrame(data)<br># Group by one column<br>grouped = df.groupby('Name')<br># Aggregate<br>grouped['Sales'].sum()      # total sales per person<br>grouped['Sales'].mean()     # average sales per person<br>grouped['Sales'].max()      # max sales per person<br>grouped['Sales'].count()    # count per person<br># Multiple aggregations<br>grouped['Sales'].agg(['sum', 'mean', 'max', 'min'])<br># Aggregate multiple columns<br>**----- End of picture text -----**<br>


**==> picture [468 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
grouped[['Sales','Cost']].sum()<br># Different agg per column<br>grouped.agg({'Sales': 'sum', 'Cost': 'mean'})<br># Group by multiple columns<br>df.groupby(['Name', 'Month'])['Sales'].sum()<br># Named aggregations (modern syntax)<br>df.groupby('Name').agg(<br>    Total_Sales=('Sales', 'sum'),<br>    Avg_Cost=('Cost', 'mean')<br>)<br># Transform — returns same-size output (useful for new columns)<br>df['Sales_Mean'] = df.groupby('Name')['Sales'].transform('mean')<br># Filter groups<br>df.groupby('Name').filter(lambda g: g['Sales'].sum() > 200)<br>**----- End of picture text -----**<br>


## **SECTION 10: Merging & Joining DataFrames** 

## **Combining DataFrames** 

## **pd.concat() — Stack DataFrames** 

**==> picture [468 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
df1 = pd.DataFrame({'A':[1,2], 'B':[3,4]})<br>df2 = pd.DataFrame({'A':[5,6], 'B':[7,8]})<br>pd.concat([df1, df2])              # stack vertically (default axis=0)<br>pd.concat([df1, df2], ignore_index=True)  # reset index<br>df3 = pd.DataFrame({'C':[9,10], 'D':[11,12]})<br>pd.concat([df1, df3], axis=1)      # stack horizontally<br>**----- End of picture text -----**<br>


## **pd.merge() — SQL-style Joins** 

merge() combines DataFrames on a common key column, similar to SQL JOINs. 

**==> picture [468 x 489] intentionally omitted <==**

**----- Start of picture text -----**<br>
employees = pd.DataFrame({<br>    'emp_id': [1, 2, 3, 4],<br>    'name':   ['Alice','Bob','Charlie','Diana'],<br>    'dept_id':[10, 20, 10, 30]<br>})<br>departments = pd.DataFrame({<br>    'dept_id': [10, 20, 40],<br>    'dept':    ['Engineering','Marketing','Finance']<br>})<br># INNER JOIN — only matching rows<br>pd.merge(employees, departments, on='dept_id')<br># Result: Alice, Bob, Charlie (not Diana — dept 30 not in depts)<br># LEFT JOIN — all left rows, NaN where no match<br>pd.merge(employees, departments, on='dept_id', how='left')<br># Diana included with NaN dept<br># RIGHT JOIN — all right rows<br>pd.merge(employees, departments, on='dept_id', how='right')<br># OUTER JOIN — all rows from both<br>pd.merge(employees, departments, on='dept_id', how='outer')<br># Join on different column names<br>pd.merge(df1, df2, left_on='emp_id', right_on='id')<br># Join on index<br>pd.merge(df1, df2, left_index=True, right_index=True)<br># Suffix for overlapping column names<br>pd.merge(df1, df2, on='id', suffixes=('_left','_right'))<br>**----- End of picture text -----**<br>


## **DataFrame.join()** 

```
# Join on index (simpler than merge for index-based joins)
df1.join(df2)                 # left join by default
df1.join(df2, how='inner')    # inner join
df1.join(df2, how='outer')    # outer join
```

## **SECTION 11: Pivot Tables & Cross-tabulation** 

## **Pivot Tables** 

**==> picture [468 x 378] intentionally omitted <==**

**----- Start of picture text -----**<br>
data = {<br>    'Employee': ['Alice','Alice','Bob','Bob','Charlie'],<br>    'Month':    ['Jan','Feb','Jan','Feb','Jan'],<br>    'Sales':    [100, 150, 200, 180, 90],<br>    'Cost':     [40, 55, 70, 65, 30]<br>}<br>df = pd.DataFrame(data)<br># Simple pivot table<br>pd.pivot_table(df,<br>               values='Sales',<br>               index='Employee',<br>               columns='Month',<br>               aggfunc='sum')<br># Multiple values and functions<br>pd.pivot_table(df,<br>               values=['Sales','Cost'],<br>               index='Employee',<br>               columns='Month',<br>               aggfunc={'Sales':'sum', 'Cost':'mean'},<br>               fill_value=0)   # fill NaN with 0<br># Cross-tabulation (frequency count)<br>pd.crosstab(df['Employee'], df['Month'])<br>pd.crosstab(df['Employee'], df['Month'], values=df['Sales'], aggfunc='sum')<br>**----- End of picture text -----**<br>


## **Stack and Unstack** 

```
# stack — moves column index to row index (wide -> long)
```

```
pivoted = pd.pivot_table(df, values='Sales', index='Employee', columns='Month',
aggfunc='sum')
pivoted.stack()   # converts to long format
# unstack — opposite (long -> wide)
long_df.unstack()
```

## **SECTION 12: Apply, Map & Transform** 

## **Applying Custom Functions** 

## **Series.apply() — Apply function to each element** 

```
s = pd.Series([1, 4, 9, 16])
s.apply(np.sqrt)              # [1. 2. 3. 4.]
s.apply(lambda x: x**2)       # [1 16 81 256]
s.apply(lambda x: 'big' if x > 8 else 'small')
```

## **DataFrame.apply() — Apply function along axis** 

```
df = pd.DataFrame({'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]})
# Apply to each COLUMN (axis=0, default)
df.apply(np.sum)        # sum of each column
df.apply(np.mean)       # mean of each column
# Apply to each ROW (axis=1)
df.apply(np.sum, axis=1)     # sum of each row
df.apply(lambda row: row.max() - row.min(), axis=1)  # range per row
# Apply function that returns a Series
def stats(col):
    return pd.Series({'mean': col.mean(), 'std': col.std()})
df.apply(stats)  # returns DataFrame with mean and std rows
```

## **map() — Element-wise mapping** 

```
s = pd.Series(['cat', 'dog', 'bird'])
# With dictionary
s.map({'cat': 'feline', 'dog': 'canine', 'bird': 'avian'})
# With function
s.map(str.upper)   # ['CAT', 'DOG', 'BIRD']
```

```
# For DataFrame: use applymap / map (Pandas 2.x)
df.map(lambda x: x * 2)   # element-wise on whole DataFrame
```

## **SECTION 13: Working with Dates & Times** 

## **DateTime in Pandas** 

**==> picture [468 x 337] intentionally omitted <==**

**----- Start of picture text -----**<br>
# Parse dates<br>df['Date'] = pd.to_datetime(df['Date'])<br>df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')<br># Date range<br>dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')  # daily<br>dates = pd.date_range('2024-01-01', periods=12, freq='M')     # monthly<br># DateTime attributes via .dt accessor<br>df['Year']    = df['Date'].dt.year<br>df['Month']   = df['Date'].dt.month<br>df['Day']     = df['Date'].dt.day<br>df['Weekday'] = df['Date'].dt.dayofweek  # 0=Monday<br>df['Quarter'] = df['Date'].dt.quarter<br>df['DayName'] = df['Date'].dt.day_name()   # 'Monday'<br># Filter by date<br>df[df['Date'] > '2024-06-01']<br>df[df['Date'].dt.month == 6]<br># Time difference<br>df['Days_Since'] = pd.Timestamp.now() - df['Date']<br>df['Days_Since'].dt.days   # number of days as int<br>**----- End of picture text -----**<br>


**==> picture [469 x 29] intentionally omitted <==**

**----- Start of picture text -----**<br>
SECTION 14: Reading & Writing Files<br>**----- End of picture text -----**<br>


## **I/O Operations** 

## **CSV** 

```
# Read CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', header=0)          # row 0 is header
df = pd.read_csv('data.csv', index_col='Name')  # set index
df = pd.read_csv('data.csv', usecols=['A','B']) # select columns
df = pd.read_csv('data.csv', nrows=100)         # read first 100 rows
df = pd.read_csv('data.csv', sep=';')           # semicolon-separated
df = pd.read_csv('data.csv', na_values=['NA','?']) # custom NaN markers
# Write CSV
df.to_csv('output.csv')             # includes index
df.to_csv('output.csv', index=False) # no index column
df.to_csv('output.csv', columns=['Name','Score'])  # select cols
```

## **Excel** 

```
# Read Excel
df = pd.read_excel('data.xlsx')
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # first sheet
# Read all sheets
sheets = pd.read_excel('data.xlsx', sheet_name=None)  # dict of dfs
# Write Excel
df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='Results', index=False)
# Write multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

## **JSON & Other** 

```
# JSON
```

```
df = pd.read_json('data.json')
df.to_json('output.json', orient='records')
# SQL
import sqlite3
```

```
conn = sqlite3.connect('mydb.sqlite')
```

```
df = pd.read_sql('SELECT * FROM users', conn)
```

```
df.to_sql('users', conn, if_exists='replace', index=False)
```

## **SECTION 15: Useful Functions & Tips** 

## **Commonly Used Functions** 

```
# Value counts — frequency of each value
```

```
df['City'].value_counts()           # sorted by freq desc
```

```
df['City'].value_counts(normalize=True)  # as proportions
```

```
# Correlation
```

```
df.corr()                 # pairwise correlation matrix
```

```
df['A'].corr(df['B'])     # correlation between two columns
```

```
# Duplicates
```

```
df.duplicated()           # boolean Series — True where duplicate
```

```
df.duplicated(subset=['Name'])  # check duplicates in one col
```

```
df.drop_duplicates()      # remove duplicate rows
```

```
# Sample
```

```
df.sample(3)              # 3 random rows
```

```
df.sample(frac=0.5)       # 50% of rows randomly
```

```
df.sample(frac=1).reset_index(drop=True)  # shuffle DataFrame
```

```
# Clip — limit values to a range
```

```
df['Score'].clip(0, 100)  # cap at 0 and 100
```

```
# cut — bin continuous data into categories
```

```
df['Grade'] = pd.cut(df['Score'],
```

```
                     bins=[0,60,75,90,100],
```

```
                     labels=['D','C','B','A'])
```

```
# qcut — quantile-based binning
```

```
pd.qcut(df['Score'], q=4, labels=['Q1','Q2','Q3','Q4'])
```

```
# Percentile / quantile
```

```
df['Score'].quantile(0.75)   # 75th percentile
```

```
# nlargest / nsmallest
```

```
df.nlargest(3, 'Score')      # top 3 rows by Score
df.nsmallest(2, 'Age')       # bottom 2 rows by Age
# Check for null
df['Score'].isnull().any()   # True if any NaN
df['Score'].notnull().all()  # True if none are NaN
# Cumulative operations
df['Score'].cumsum()    # cumulative sum
df['Score'].cumprod()   # cumulative product
df['Score'].cummax()    # cumulative maximum
df['Score'].diff()      # difference from previous row
```

## **QUICK REFERENCE CHEAT SHEET** 

## **Pandas Quick Reference** 

|**Operation**|**Code / Description**|
|---|---|
|**Create Series**|`pd.Series([1,2,3])`|
|**Create DataFrame**|`pd.DataFrame({'A':[1,2],'B':[3,4]})`|
|**Read CSV**|`pd.read_csv('file.csv')`|
|**First/last N rows**|`df.head(N) / df.tail(N)`|
|**Shape & dtypes**|`df.shape / df.dtypes / df.info()`|
|**Select column**|`df['col'] or df[['col1','col2']]`|
|**Select by label**|`df.loc[row, col]`|
|**Select by position**|`df.iloc[row, col]`|
|**Filter rows**|`df[df['col'] > value]`|
|**Multiple conditions**|`df[(cond1) & (cond2)]`|
|**Add column**|`df['new'] = expression`|
|**Drop column**|`df.drop('col', axis=1)`|
|**Rename columns**|`df.rename(columns={'old':'new'})`|
|**Sort values**|`df.sort_values('col', ascending=False)`|
|**Handle NaN**|`df.fillna(0) / df.dropna()`|
|**GroupBy**|`df.groupby('col')['val'].sum()`|
|**Merge (join)**|`pd.merge(df1, df2, on='key', how='left')`|
|**Concat**|`pd.concat([df1, df2], ignore_index=True)`|
|**Apply function**|`df['col'].apply(lambda x: ...)`|
|**String ops**|`df['col'].str.upper() / .contains()`|



|**Pivot table**|`pd.pivot_table(df, values=, index=, columns=)`|
|---|---|
|**Duplicates**|`df.drop_duplicates()`|
|**Value counts**|`df['col'].value_counts()`|



