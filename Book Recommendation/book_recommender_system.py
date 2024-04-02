# Importing necessary libraries
import numpy as np
import pandas as pd

# Adjusting display options for pandas
pd.set_option("display.max_colwidth", 1000)
# Data visualization libraries
import seaborn as sns

sns.set_style("white")
import matplotlib.pyplot as plt

# Library for working with images
from PIL import Image

# Libraries for word cloud creation
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS

# Setting plot size
plt.rcParams["figure.figsize"] = (8, 8)
# Importing libraries for displaying images and HTML content
from IPython.display import Image as IPImage, HTML as IPHTML

from tabulate import tabulate

# #loading the required datasets
books_df = pd.read_csv("Books.csv")
rating_df = pd.read_csv("Users.csv")
users_df = pd.read_csv("Ratings.csv")

# Displaying the first 10 rows of the users_df DataFrame
users_df.head(10)

# Displaying the first 10 rows of the books_df DataFrame
books_df.head(10)

# Displaying the first 10 rows of the rating_df DataFrame
rating_df.head(10)

# Printing the shapes of all the datasets: Users, Books, and Ratings
print(
    "Shape of Users: {}, Books: {} and Ratings: {}".format(
        users_df.shape, books_df.shape, rating_df.shape
    )
)

# Printing information about the books_df DataFrame
# This includes details such as the column names, data types, and memory usage
print(books_df.info())

# Getting the DataFrame information
info_table = books_df.info()

# Convert the DataFrame information to a string with tabulate
table_str = tabulate(info_table, headers="keys", tablefmt="grid")

# Printing the table
print(table_str)

# Getting the summary statistics of the books_df DataFrame
summary_stats = books_df.describe()

# Convert the summary statistics to a string with tabulate
table_str = tabulate(summary_stats, headers="keys", tablefmt="grid")

# Printing the table
print(table_str)

# Missing value percentage
from tabulate import tabulate

# Calculate the percentage of missing values in each column
missing_percentage = books_df.isnull().sum() / len(books_df) * 100

# Convert the missing percentage to a DataFrame for tabulation
missing_percentage_df = missing_percentage.to_frame().reset_index()
missing_percentage_df.columns = ["Column", "Missing Percentage"]

# Convert the DataFrame to a string with tabulate
table_str = tabulate(
    missing_percentage_df, headers="keys", tablefmt="grid", showindex=False
)

# Print the table
print(table_str)

# Checking  for  null value in book author
books_df[books_df["Book-Author"].isna()]

# Filling the null value
books_df.loc[187689, "Book-Author"] = "Larissa Anne Downes"

# Checking null values in publisher
books_df[books_df["Publisher"].isna()]

# Replacing NaNs with correct  values
books_df.loc[128890, "Publisher"] = "Mundania Press LLC"
books_df.loc[129037, "Publisher"] = "Bantam"

# insepcting the values in year of publication
books_df["Year-Of-Publication"].unique()

"""Name of few publication companies have been incorrectly put in this column.There are values such as 0 ,2024,2030 etc. which is also not possible .Let's rectify these mistakes"""

# correcting this error
books_df[books_df["Year-Of-Publication"] == "DK Publishing Inc"]

# on searching for these  books we came to know about its authors
# ISBN '078946697X'
books_df.loc[books_df.ISBN == "078946697X", "Year-Of-Publication"] = 2000
books_df.loc[books_df.ISBN == "078946697X", "Book-Author"] = "Michael Teitelbaum"
books_df.loc[books_df.ISBN == "078946697X", "Publisher"] = "DK Publishing Inc"
books_df.loc[books_df.ISBN == "078946697X", "Book-Title"] = (
    "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
)

# ISBN '0789466953'
books_df.loc[books_df.ISBN == "0789466953", "Year-Of-Publication"] = 2000
books_df.loc[books_df.ISBN == "0789466953", "Book-Author"] = "James Buckley"
books_df.loc[books_df.ISBN == "0789466953", "Publisher"] = "DK Publishing Inc"
books_df.loc[books_df.ISBN == "0789466953", "Book-Title"] = (
    "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
)

# checking the rows having 'Gallimard' as yearOfPublication
books_df.loc[books_df["Year-Of-Publication"] == "Gallimard", :]

books_df.loc[books_df.ISBN == "2070426769", "Year-Of-Publication"] = 2003
books_df.loc[books_df.ISBN == "2070426769", "Book-Author"] = (
    "Jean-Marie Gustave Le ClÃ?Â©zio"
)
books_df.loc[books_df.ISBN == "2070426769", "Publisher"] = "Gallimard"
books_df.loc[books_df.ISBN == "2070426769", "Book-Title"] = (
    "Peuple du ciel, suivi de 'Les Bergers"
)

# Checking if the corrections are in place
books_df.loc[books_df["ISBN"].isin(["2070426769", "0789466953", "078946697X"])]

# changing dtype of year of publication
books_df["Year-Of-Publication"] = books_df["Year-Of-Publication"].astype(int)

# something is off about years of publication like:
books_df[
    (books_df["Year-Of-Publication"] > 0) & (books_df["Year-Of-Publication"] < 1800)
]

# replacing with correct  values
books_df.loc[books_df.ISBN == "	9643112136", "Year-Of-Publication"] = 2010
books_df.loc[books_df.ISBN == "964442011X", "Year-Of-Publication"] = 1991

# Sustituting np.Nan in rows with year=0 or  greater than the current year,2022.
books_df.loc[
    (books_df["Year-Of-Publication"] > 2022) | (books_df["Year-Of-Publication"] == 0),
    "Year-Of-Publication",
] = np.NAN

# replacing NaN values with median value of Year-Of-Publication
books_df["Year-Of-Publication"].fillna(
    int(books_df["Year-Of-Publication"].median()), inplace=True
)

books_df["Book-Author"].value_counts()

books_df["Publisher"].value_counts()

# Uppercasing the ISBN numbers
books_df["ISBN"] = books_df["ISBN"].str.upper()

books_df[books_df["Book-Title"] == "Jasper"]

# Checking for duplicates in books_df
books_df[books_df.duplicated()]

# Dropping the rows with the entire column values are duplicated
books_df.drop_duplicates(keep="first", inplace=True)
books_df.reset_index(drop=True, inplace=True)

# displaying the top 10 and bottom 10 rows of the dataframe
pd.concat([users_df.head(10), users_df.tail(10)], axis=0)

# inspecting the columns in users_df
users_df.info()

# Violin plot showing distribution of ages
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.violinplot(x="Age", data=users_df, inner="quartile")

# Boxplot of Age column
plt.subplot(1, 2, 2)
sns.boxplot(x="Age", data=users_df)

plt.show()

# replacing the outliers in 'Age' with NaN value
users_df.loc[(users_df["Age"] > 95) | (users_df["Age"] < 4), "Age"] = np.nan

# Displaying the current number of missing values in  'Age' column
print("The number of missing values is ", users_df["Age"].isnull().sum())
# Imputing such a large amount of null values will mean/mode/median will drastically change the distribution
users_df["Age"].describe()

# create a normal distribution pd.Series to fill Nan values with
normal_age_series = pd.Series(
    np.random.normal(
        loc=users_df.Age.mean(),
        scale=users_df.Age.std(),
        size=users_df[users_df.Age.isna()]["User-ID"].count(),
    )
)

# take the absolute value of temp_age_series
abs_age_series = round(np.abs(normal_age_series), 0)

# sort users df so as NaN values in age to be first and reset index to match with index of abs_age_series. Then using fillna()
users_df = users_df.sort_values("Age", na_position="first").reset_index(drop=True)
users_df.Age.fillna(abs_age_series, inplace=True)

# after imputation
users_df.Age.describe()

# percentage of missing values in 'Age' column
(users_df["Age"].isnull().sum() / len(users_df)) * 100


def age_group(age):
    """
    defines the age group of users
    """
    if age < 13:
        x = "Children"
    elif age >= 13 and age < 18:
        x = "Teens"
    elif age >= 18 and age < 36:
        x = "Youth"
    elif age >= 36 and age < 56:
        x = "Middle aged adults"
    else:
        x = "Elderly"
    return x


users_df["Age_group"] = users_df["Age"].apply(lambda x: age_group(x))

# Age distribution of users
sns.countplot(users_df["Age_group"], palette="Set2")

# number of unique values in 'Location'
users_df["Location"].nunique()

# extracting the country names from users_df
for i in users_df:
    users_df["Country"] = users_df.Location.str.extract(r"\,+\s?(\w*\s?\w*)\"*$")

# Displaying the country names
set(users_df["Country"])

# correcting the mispelled country names
users_df.loc[
    users_df["Country"].isin(["australii", "autralia", "western australia"]), "Country"
] = "australia"
users_df.loc[
    users_df["Country"].isin(
        [
            "unite states",
            "01776",
            "02458",
            "19104",
            "23232",
            "30064",
            "85021",
            "87510",
            "united sates",
            "united staes",
            "united state",
            "united statea",
            "united stated",
            "america" "united stated of america",
            "united states",
            "united states of america",
            "us",
            "us of a",
            "us virgin islands",
            "usa  canada",
            "usa currently living in england",
            "uusa",
            "usaa",
            "wonderful usa",
            "california",
            "orange co",
        ]
    ),
    "Country",
] = "usa"
users_df.loc[
    users_df["Country"].isin(
        ["united kindgdom", "united kindgonm", "united kingdom", "u k"]
    ),
    "Country",
] = "uk"
users_df.loc[
    users_df["Country"].isin(
        [
            "the philippines",
            "philippines",
            "philippinies",
            "phillipines",
            "phils",
            "phippines",
        ]
    ),
    "Country",
] = "philippines"
users_df.loc[
    users_df["Country"].isin(
        [
            "",
            "xxxxxx",
            "universe",
            "nowhere",
            "x",
            "y",
            "a",
            "öð¹ú",
            "the",
            "unknown",
            np.nan,
            "n/a",
            "aaa",
            "z",
            "somewherein space",
        ]
    ),
    "Country",
] = "others"
users_df.loc[users_df["Country"].isin(["italia", "italien", "itlay"]), "Country"] = (
    "italy"
)
users_df.loc[
    users_df["Country"].isin([" china öð¹ú", "chinaöð¹ú", "chian"]), "Country"
] = "china"
users_df["Country"].replace(
    [
        "the gambia",
        "the netherlands",
        "geermany",
        "srilanka",
        "saudia arabia",
        "brasil",
        "_ brasil",
        "indiai",
        "malaysian",
        "hongkong",
        "russian federation",
    ],
    [
        "gambia",
        "netherlands",
        "germany",
        "sri lanka",
        "saudi arabia",
        "brazil",
        "brazil",
        "india",
        "malaysia",
        "hong kong",
        "russia",
    ],
    inplace=True,
)

# Converting the country names to uppercase
users_df["Country"] = users_df["Country"].str.upper()

# Dropping the column 'Location'
users_df.drop("Location", axis=1, inplace=True)

users_df.columns

users_df.Country.value_counts()

# displaying the first 5 rows
rating_df.head()

rating_df.info()

# checking null values
rating_df.isna().sum()

# checking for unique user ids and isbn values
print(
    "Number of unique user ids is {} and ISBN no. is {}".format(
        rating_df["User-ID"].nunique(), rating_df["ISBN"].nunique()
    )
)

# making all the ISBN no. uppercase
rating_df["ISBN"].apply(lambda x: x.upper())

# checking for duplicates
rating_df[rating_df.duplicated()].sum()

# lets see if all the books in rating_df are also in books_df
rating_df_new = rating_df[rating_df["ISBN"].isin(books_df["ISBN"])]

print(
    "Shape of rating_df: {} and rating_df_new: {}".format(
        rating_df.shape, rating_df_new.shape
    )
)

# book ratings
rating_df_new["Book-Rating"].value_counts().reset_index()

# most popular books
rating_df_new.groupby("ISBN")["Book-Rating"].count().reset_index().sort_values(
    by="Book-Rating", ascending=False
)[:10]

explicit_rating = rating_df_new[rating_df_new["Book-Rating"] != 0]
implicit_rating = rating_df_new[rating_df_new["Book-Rating"] == 0]
print(
    "Shape of explicit rating: {} and implicit rating: {}".format(
        explicit_rating.shape, implicit_rating.shape
    )
)

# most purchased books including the implicitely rated books
rating_df_new.groupby("ISBN")["User-ID"].count().reset_index().sort_values(
    by="User-ID", ascending=False
)[:10]["ISBN"].values

# getting the book names corresponding to these ISBNs
isbn_nums = [
    "0971880107",
    "0316666343",
    "0385504209",
    "0060928336",
    "0312195516",
    "044023722X",
    "0142001740",
    "067976402X",
    "0671027360",
    "0446672211",
]
books_df[books_df["ISBN"].isin(isbn_nums)]

# most popular explicitely rated books
explicit_rating.groupby("ISBN")["Book-Rating"].count().reset_index().sort_values(
    by="Book-Rating", ascending=False
)[:10]

# getting the book names corresponding to these ISBNs
isbn_nums = ["0316666343", "0971880107", "0385504209", "0312195516", "0060928336"]
books_df[books_df["ISBN"].isin(isbn_nums)]

# for the rating dataset, we are only taking the explicit rating dataset

books_rating_df = pd.merge(books_df, explicit_rating, on="ISBN", how="inner")
df = pd.merge(books_rating_df, users_df, on="User-ID", how="inner")

# shape of the merged dataframe 'df'
df.shape

# displaying the top 3 rows of df
df.head(3)

df.info()

# Number of users who have rated the books
df["User-ID"].nunique()

# Number of books in the dataframe
df["ISBN"].nunique()

"""## **Exploratory Data Analysis**"""

# Age distribution of users
age_df = users_df[users_df["User-ID"].isin(list(df["User-ID"].unique()))]
sns.distplot(age_df.Age)

# Rating distribution for explicit rating

sns.countplot(x="Book-Rating", data=df, palette="Set2")

fig = plt.figure(figsize=(12, 8))
# Authored the most number of  books
plt.subplot(1, 2, 1)
popular_authors = books_df.groupby("Book-Author")["Book-Title"].count().nlargest(10)
sns.barplot(x=popular_authors.values, y=popular_authors.index, palette="Paired")
plt.title("Top ten writers in terms of number of books published")

# Best selling authors
plt.subplot(1, 2, 2)
best_selling_authors = df.groupby("Book-Author")["User-ID"].nunique().nlargest(10)
sns.barplot(
    x=best_selling_authors.values, y=best_selling_authors.index, palette="Paired"
)
plt.title("Best selling authors")
plt.tight_layout()

df.groupby("Book-Title")["User-ID"].count().nlargest(10)

# Top selling books
most_purchased_books = df.groupby("Book-Title")["User-ID"].nunique().nlargest(10)
sns.barplot(
    x=most_purchased_books.values, y=most_purchased_books.index, palette="Paired"
)
plt.title("Top selling books")

# Top-rated books
top_rated_books = df.groupby("Book-Title")["Book-Rating"].sum().nlargest(10)
sns.barplot(x=top_rated_books.values, y=top_rated_books.index, palette="Paired")
plt.title("Top rated books")

fig = plt.figure(figsize=(20, 14))
i = 1
for group in ["Children", "Teens", "Youth", "Middle aged adults", "Elderly"]:
    age_df = (
        df.loc[df["Age_group"] == group]
        .groupby(["Book-Title"])
        .agg(No_of_users=("User-ID", "nunique"), total_rating=("Book-Rating", "sum"))
        .reset_index()
    )
    plt.subplot(5, 2, i)
    age_df.sort_values(by="No_of_users", ascending=False, inplace=True)
    sns.barplot(x="No_of_users", y="Book-Title", palette="Paired", data=age_df.head(5))
    plt.title("Top 5 Popular books among  {}".format(group), size=16)
    i += 1
    plt.subplot(5, 2, i)
    age_df.sort_values(by="total_rating", ascending=False, inplace=True)
    sns.barplot(x="total_rating", y="Book-Title", palette="Set2", data=age_df.head(5))
    plt.title("Top rated books by {} ".format(group), size=16)
    i += 1

plt.tight_layout()
