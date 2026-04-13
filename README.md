Experiment No. 15

Name: Pranav Menon

PRN No.: 25070123085

Batch: ENTC - B1

Aim: To perform Data Normalization and Data Type Conversion using Python.


THEORY

1) Introduction to Data Normalization

Data normalization is the process of rescaling numerical features so that they fall within
a common range or follow a standard distribution. Raw numerical data often spans vastly
different scales — for example, a Price column may contain values in the tens of thousands
while a Discount column contains values between 0 and 100. When such features are used
together in machine learning algorithms (particularly distance-based algorithms like k-nearest
neighbours, k-means clustering, or gradient descent-based models), columns with larger
magnitudes dominate computations and can cause biased, incorrect, or slow-converging results.

Normalization eliminates this scale disparity without distorting the relative relationships
between values, ensuring every feature contributes fairly to an analysis.


2) Dataset Creation

Create a 6-row, 4-column DataFrame. Price spans from ₹2,000 to ₹55,000 — a factor
of 27.5x difference. Discount spans only 0–20. Without normalization, Price would
completely dominate any distance or gradient calculation.

3) Min-Max Normalization — Single Column

Min-Max Normalization rescales all values in a column to lie within [0, 1]. The minimum
value maps to 0, the maximum to 1, and all others are proportionally placed between them.

Formula:
    x_normalized = (x - x_min) / (x_max - x_min)

Implementation:
    df['Price_MinMax'] = (df['Price'] - df['Price'].min()) / \
                         (df['Price'].max() - df['Price'].min())
    print(df[['Product','Price','Price_MinMax']])

df['Price'].min() returns the minimum value (2000). df['Price'].max() returns the maximum
(55000). The denominator (max - min) is 53000. Each Price value is shifted by subtracting
the minimum, then scaled by dividing by the range:
  - Laptop (55000): (55000 - 2000) / 53000 = 1.000
  - Mobile (20000): (20000 - 2000) / 53000 ≈ 0.340
  - Headphones (2000): (2000 - 2000) / 53000 = 0.000

Selecting only specific columns for printing — df[['Product','Price','Price_MinMax']] —
uses double square brackets to pass a list of column names, returning a DataFrame with
just those three columns.


4) Min-Max Normalization — Multiple Columns Simultaneously

    cols = ['Price', 'Units Sold', 'Discount']
    df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    print(df_norm)

df[cols] selects a sub-DataFrame with only the three specified columns. df[cols].min() and
df[cols].max() compute the column-wise minimum and maximum, returning a Series. Pandas
broadcasts these Series across the rows automatically (column-by-column), so each column is
normalized independently using its own min and max. The result df_norm is a new 6×3
DataFrame with all values in [0, 1].

This is more concise and efficient than applying the formula separately for each column.
The same approach was applied to the amazon dataset:
    cols = ['Rating', 'Reviews', 'Units_Sold']
    df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    print(df_norm)


5) Z-Score Normalization (Standardization)

Z-Score Normalization transforms values so that the resulting column has a mean of 0 and a
standard deviation of 1. Each value is replaced by the number of standard deviations it
lies above or below the column mean.

Formula:
    z = (x - mean) / std

Implementation:
    df['Units ZScore'] = (df['Units Sold'] - df['Units Sold'].mean()) / \
                         df['Units Sold'].std()
    print(df[['Product', 'Units Sold', 'Units ZScore']])

df['Units Sold'].mean() computes the arithmetic mean of the column (ignoring NaN).
df['Units Sold'].std() computes the sample standard deviation (using Bessel's correction,
dividing by n-1 by default). Positive Z-scores indicate values above the mean; negative
Z-scores indicate below-mean values.

Example interpretation:
  - If mean = 44.17 and std ≈ 23.67, then Headphones (80 units) has Z = (80 - 44.17) /
    23.67 ≈ 1.51, meaning it is 1.51 standard deviations above the average unit count.
  - Camera (15 units) would have a negative Z-score.

Unlike Min-Max, Z-Score does not bound the output to [0, 1]. Values can range freely,
which is appropriate for algorithms that assume normally distributed features.

The same formula was applied to the amazon dataset:
    df['Units ZScore'] = (df['Units_Sold'] - df['Units_Sold'].mean()) / \
                         df['Units_Sold'].std()


6) Decimal Scaling

Decimal Scaling normalizes values by dividing by a power of 10 chosen to be large enough
that the maximum absolute value becomes less than 1 (or close to 0).

Rule: divide by 10^j where j = ceil(log10(max(|x|))).

In practice, an appropriate power of 10 is selected based on knowledge of the column's
magnitude:

    df['Price_Decimal'] = df['Price'] / 100000
    print(df[['Product','Price','Price_Decimal']])

    df['Units_Decimal'] = df['Units Sold'] / 100
    print(df[['Product','Price','Units_Decimal']])

    df['Discounts_Decimal'] = df['Discount'] / 100
    print(df[['Product','Price','Discounts_Decimal']])

For Price (max = 55,000): dividing by 100,000 maps the range to approximately [0.02, 0.55].
For Units Sold (max = 80): dividing by 100 maps to [0.15, 0.80].
For Discount (max = 20): dividing by 100 maps to [0.00, 0.20].

Decimal Scaling is the simplest normalization method, easy to reverse (just multiply back),
and interpretable when the column's scale is well understood. The same was applied to the
amazon dataset: df['Units_Decimal'] = df['Units_Sold'] / 100.


7) Introduction to Categorical Encoding

Machine learning algorithms require all input features to be numerical. Categorical columns
(strings) must be converted to numbers before they can be used as model features. This
process is called categorical encoding or data type conversion.

8) Label Encoding — sklearn.preprocessing.LabelEncoder

Label Encoding assigns a unique integer to each distinct category in a column. The mapping
is determined alphabetically by default.

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

The LabelEncoder is instantiated once and reused for each column via fit_transform():

    df['Gender_Label'] = le.fit_transform(df['Customer_Gender'])

fit_transform() both learns the mapping (fit) and applies it (transform) in a single step.
For Customer_Gender: 'Female' → 0, 'Male' → 1.

    df['Payment_Method_Label'] = le.fit_transform(df['Payment_Method'])

For Payment_Method (alphabetically): 'COD' → 0, 'Credit Card' → 1, 'Debit Card' → 2,
'UPI' → 3.

    df['Product_Category_Label'] = le.fit_transform(df['Product_Category'])

For Product_Category: 'Beauty' → 0, 'Clothing' → 1, 'Electronics' → 2, 'Home' → 3.

    df['City_Label'] = le.fit_transform(df['City'])

For City: 'Bangalore' → 0, 'Delhi' → 1, 'Hyderabad' → 2, 'Mumbai' → 3, 'Pune' → 4.

Each call to le.fit_transform() resets and relearns the mapping from scratch for the new
column. The labels are always integers starting from 0. This was also applied to the
Student-Dataset: df['Gender_Label'] = le.fit_transform(df['Gender']).

Limitation: Label encoding introduces a false ordinal relationship. The model may interpret
City encoded as Bangalore=0, Delhi=1 as meaning Delhi is "greater than" Bangalore, which
has no real meaning. This is why One-Hot Encoding is preferred for nominal (unordered)
categorical variables.


9) One-Hot Encoding — pd.get_dummies()

One-Hot Encoding creates a new binary (0 or 1) column for each distinct category in a
column. For a column with k unique categories, k new columns are created, each representing
one category. Each row has exactly one 1 among the new columns and 0 everywhere else.

For Payment_Method with 4 categories (COD, Credit Card, Debit Card, UPI), four new columns
are created: Payment_Method_COD, Payment_Method_Credit Card, Payment_Method_Debit Card,
Payment_Method_UPI. A row where Payment_Method was 'UPI' will have Payment_Method_UPI=1
and all other Payment_Method_* columns as 0.

pd.get_dummies() automatically removes the original categorical column and replaces it with
the binary indicator columns. This was also applied to the Student-Dataset:
    df_encoded = pd.get_dummies(df, columns=['CGPA'])
    df_encoded

Limitation: For columns with many unique values (like Order_Value, which has 6 unique
values for only 6 rows), One-Hot Encoding creates as many columns as there are unique values,
which can cause dimensionality issues in large datasets. This is sometimes called the
"dummy variable explosion."


10) Dummy Coding — pd.get_dummies() with drop_first=True

Dummy Coding is a variant of One-Hot Encoding that creates k-1 columns for k categories
instead of k. The first category (alphabetically) is used as the reference (baseline) and
is represented implicitly when all k-1 dummy columns are 0.

For Customer_Gender (Female, Male): drop_first=True drops Customer_Gender_Female.
Only Customer_Gender_Male remains (1 = Male, 0 = Female).

For City (5 categories): drops City_Bangalore. Retains City_Delhi, City_Hyderabad,
City_Mumbai, City_Pune. A row where all four are 0 is implicitly Bangalore.

The Student-Dataset version:
    df_encoded = pd.get_dummies(df, columns=['Attendance_Percentage'], drop_first=True)
    df_encoded

The drop_first=True parameter is the standard approach in regression models to avoid the
"dummy variable trap" — a condition of perfect multicollinearity where one dummy column is
a perfect linear combination of the others, making the design matrix singular and model
coefficients undefined.


11) Comparison of Normalization Techniques

Technique       | Output Range  | Robust to Outliers | Best For
Min-Max         | [0, 1]        | No                 | Neural networks, bounded algorithms
Z-Score         | Unbounded     | Moderate           | Linear models, normally distributed data
Decimal Scaling | [0, ~1]       | No                 | Simple scaling, interpretable results


12) Comparison of Encoding Techniques

Technique       | Columns Added | Ordinal Risk | Multicollinearity Risk | Best For
Label Encoding  | 1             | Yes          | No                     | Ordinal categories
One-Hot         | k             | No           | Yes (trap possible)    | Nominal categories
Dummy Coding    | k-1           | No           | No                     | Regression models

CONCLUSION

Data Normalization and Data Type Conversion using Python were successfully performed.
