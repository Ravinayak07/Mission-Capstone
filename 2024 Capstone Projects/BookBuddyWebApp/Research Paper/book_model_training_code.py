# Importing the required libraries
import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library

pd.set_option(
    "display.max_colwidth", 1000
)  # Setting maximum column width for DataFrame display

# Data visualization libraries
import seaborn as sns  # Statistical data visualization

sns.set_style("white")
import matplotlib.pyplot as plt  # Plotting library

# Word cloud creation
from PIL import Image  # Python Imaging Library
from wordcloud import WordCloud  # Word cloud generator
from wordcloud import WordCloud, STOPWORDS  # Word cloud and stop words

plt.rcParams["figure.figsize"] = (8, 8)  # Setting default figure size
from IPython.display import Image, HTML  # Displaying images in IPython

# Suppressing warnings
import warnings

warnings.simplefilter("ignore")

# Other libraries
import os  # Operating system interface
import nltk  # Natural Language Toolkit
import sklearn  # Machine learning library

# Reading datasets into DataFrames
books_df = pd.read_csv("dataset/Books.csv")  # Reading books dataset into a DataFrame
users_df = pd.read_csv("dataset/Users.csv")  # Reading users dataset into a DataFrame
rating_df = pd.read_csv(
    "dataset/Ratings.csv"
)  # Reading ratings dataset into a DataFrame

users_df.head(3)

books_df.head(3)

rating_df.head(3)

# shapes of all the datasets
print(
    "Shape of Users: {}, Books: {} and Ratings: {}".format(
        users_df.shape, books_df.shape, rating_df.shape
    )
)

books_df.head()

print("This shape of Books datasheet is : ", books_df.shape)
print("=" * 30)
print("This shape of Ratings datasheet is : ", rating_df.shape)
print("=" * 30)
print("This shape of Users datasheet is : ", users_df.shape)

print(books_df.info())

books_df.describe()

# Calculating missing value percentage in the books dataset
print(books_df.isnull().sum() / len(books_df) * 100)

# Checking for null values in the 'Book-Author' column of the books dataset
books_df[books_df["Book-Author"].isna()]

# Filling the null value in the 'Book-Author' column with a specific author name
books_df.loc[187689, "Book-Author"] = "Larissa Anne Downes"

# Checking null values in publisher
books_df[books_df["Publisher"].isna()]

# Replacing NaNs with correct  values
books_df.loc[128890, "Publisher"] = "Mundania Press LLC"
books_df.loc[129037, "Publisher"] = "Bantam"

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

# Checking for duplicates in books_df
books_df[books_df.duplicated()]

# Dropping the rows with the entire column values are duplicated
books_df.drop_duplicates(keep="first", inplace=True)
books_df.reset_index(drop=True, inplace=True)

books_df.info()

# displaying the top 10 and bottom 10 rows of the dataframe
pd.concat([users_df.head(10), users_df.tail(10)], axis=0)

# inspecting the columns in users_df
users_df.info()

"""* There are records of 278858 users in this dataframe.There are 3 columns in this dataframe.
* The 'Age' column has null values
"""

display(users_df)

# Checking for duplicates in users_df
users_df[users_df["User-ID"].duplicated()].sum()

# Percentage of missing values in users_df
print(users_df.isnull().sum() / len(users_df) * 100)

"""* The 39.7% of values in the 'Age' column are missing/NaN values"""

# summarizing data in 'Age' column
users_df["Age"].describe()

"""* The maximum value in the 'Age' column is 244. This is certainly an outlier."""

# KDE plot showing distribution of ages with different colors
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(x="Age", data=users_df, fill=True, color="orange")

# Violin plot of Age column with different colors
plt.subplot(1, 2, 2)
sns.violinplot(x="Age", data=users_df, palette="Set2")

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


# Convert non-numeric values to NaN
users_df["Age_group"] = pd.to_numeric(users_df["Age_group"], errors="coerce")

# Drop rows with NaN values (optional)
# users_df.dropna(subset=['Age_group'], inplace=True)

# Plot age distribution
sns.countplot(users_df["Age_group"], palette="Set2")

# number of unique values in 'Location'
users_df["Location"].nunique()

# extracting the country names from users_df
for i in users_df:
    users_df["Country"] = users_df.Location.str.extract(r"\,+\s?(\w*\s?\w*)\"*$")

# Displaying the country names
set(users_df["Country"])

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

"""## **Merging Datasets**
"""

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

# Countries with maximum number of users
top10 = df.groupby("Country")["User-ID"].nunique().nlargest(15)

# Creating a countplot
plt.figure(figsize=(10, 8))
sns.countplot(y="Country", data=df, order=top10.index, palette="Paired")
plt.xlabel("Number of Users")
plt.ylabel("Country")
plt.title("Top 15 Countries by Number of Users")
plt.show()

# Age distribution of users using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Age", data=age_df, color="skyblue")
plt.xlabel("Age")
plt.title("Age Distribution of Users")
plt.show()

# Rating distribution for explicit rating
rating_counts = df["Book-Rating"].value_counts()

# Keeping the top rating values and grouping the rest into an "Other" category
top_ratings = rating_counts.head(5)
other_ratings_count = rating_counts[5:].sum()

# Creating labels for the pie chart
labels = top_ratings.index.tolist()
labels.append("Other")

# Calculating values for the pie chart
values = top_ratings.tolist()
values.append(other_ratings_count)

# Creating a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Set2"),
)
plt.title("Rating Distribution for Explicit Ratings")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Authored the most number of books
popular_authors = books_df.groupby("Book-Author")["Book-Title"].count().nlargest(10)

# Best selling authors
best_selling_authors = df.groupby("Book-Author")["User-ID"].nunique().nlargest(10)

# Creating pie charts
plt.figure(figsize=(16, 8))

# Pie chart for top ten writers in terms of number of books published
plt.subplot(1, 2, 1)
plt.pie(
    popular_authors,
    labels=popular_authors.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Paired"),
)
plt.title("Top Ten Writers in Terms of Number of Books Published")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

# Pie chart for best selling authors
plt.subplot(1, 2, 2)
plt.pie(
    best_selling_authors,
    labels=best_selling_authors.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Paired"),
)
plt.title("Best Selling Authors")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

plt.tight_layout()
plt.show()

# Companies with the most number of books published
popular_publishers = books_df.groupby("Publisher")["Book-Title"].count().nlargest(10)

# Creating a point plot
plt.figure(figsize=(10, 8))
sns.pointplot(x=popular_publishers.values, y=popular_publishers.index, color="blue")
plt.xlabel("Number of Books Published")
plt.ylabel("Publisher")
plt.title("Top Ten Publishers in Terms of Number of Books Published")
plt.show()

df.groupby("Book-Title")["User-ID"].count().nlargest(10)

# Top selling books
most_purchased_books = df.groupby("Book-Title")["User-ID"].nunique().nlargest(10)

# Creating a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    most_purchased_books,
    labels=most_purchased_books.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Paired"),
)
plt.title("Top Selling Books")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Top-rated books
top_rated_books = df.groupby("Book-Title")["Book-Rating"].sum().nlargest(10)

# Creating a vertical bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x=top_rated_books.index, y=top_rated_books.values, palette="Paired")
plt.xlabel("Book Title")
plt.ylabel("Total Rating")
plt.title("Top Rated Books")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

fig = plt.figure(figsize=(20, 14))
i = 1
for group in ["Children", "Teens", "Youth", "Middle aged adults", "Elderly"]:
    age_df = (
        df.loc[df["Age_group"] == group]
        .groupby(["Book-Title"])
        .agg(No_of_users=("User-ID", "nunique"), total_rating=("Book-Rating", "sum"))
        .reset_index()
    )
    if not age_df.empty:
        plt.subplot(5, 2, i)
        age_df.sort_values(by="No_of_users", ascending=False, inplace=True)
        sns.barplot(
            x="No_of_users", y="Book-Title", palette="Paired", data=age_df.head(5)
        )
        plt.title("Top 5 Popular books among {}".format(group), size=16)
        i += 1
        plt.subplot(5, 2, i)
        age_df.sort_values(by="total_rating", ascending=False, inplace=True)
        sns.barplot(
            x="total_rating", y="Book-Title", palette="Set2", data=age_df.head(5)
        )
        plt.title("Top rated books by {}".format(group), size=16)
        i += 1
plt.tight_layout()
plt.show()

from IPython.display import Image, HTML  # Displaying images

"""## **Popularity based Recommender System**
"""

# finding the average rating and number of votes received by books
df_relevant_data = df.groupby(["Book-Title", "Book-Author"], as_index=False).agg(
    avg_rating=("Book-Rating", "mean"), ratings_count=("Book-Rating", "count")
)
v = df_relevant_data["ratings_count"]
R = df_relevant_data["avg_rating"]
C = df_relevant_data["avg_rating"].mean()
m = int(
    df_relevant_data["ratings_count"].quantile(0.90)
)  # minimum number of votes to be listed
print(
    f"The average rating of all the books is {C} and the minimum number of votes required by the books to be listed is {m}  "
)

# Calculating weighted average rating of the books
df_relevant_data["weighted_average"] = round(((R * v) + (C * m)) / (v + m), 2)

df_relevant_data.sort_values(by="weighted_average", ascending=False).head(10)

"""
## **Author based recommender system**
"""


def author_based(book_title, number, df_relevant_data=df_relevant_data):
    """
    To recommend books from the same author as the book entered by the user
    """
    author = df_relevant_data.loc[df_relevant_data["Book-Title"] == book_title][
        "Book-Author"
    ].unique()[0]
    author_df = df_relevant_data.loc[
        (df_relevant_data["Book-Author"] == author)
    ].sort_values(by="weighted_average", ascending=False)
    print(f"The author of the book {book_title} is {author}\n")
    print(f"Here are the top {number} books from the same author\n")
    top_rec = author_df.loc[
        (author_df["Book-Title"] != book_title), ["Book-Title", "weighted_average"]
    ].head(number)
    return top_rec


# get book name and number of books to recommend
book_title = "Harry Potter and the Chamber of Secrets (Book 2)"
number = 5
author_based(book_title, number)
# top_recommendations from the same author

"""# **Collaborative filtering**
"""

## **Model Based Approach**

## 1. **Singular Value Decomposition**

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import random
import math

"""###**Filtering the number of books and users**"""

# Filtering books with more than 5 reviews

print(
    "The number of books that are explicitely rated are",
    explicit_rating["ISBN"].nunique(),
)
ratings_count_df = (
    explicit_rating.groupby("ISBN")["User-ID"]
    .count()
    .to_frame("No-of-rated-users")
    .reset_index()
)
selected_books = list(
    ratings_count_df.loc[ratings_count_df["No-of-rated-users"] > 5, "ISBN"].unique()
)
print("Number of  books rated by atleast 5 users:", len(selected_books))
filter_df = explicit_rating.loc[explicit_rating["ISBN"].isin(selected_books)]

# keeping books with selected users
print(
    "The number of users who have explicitely rated books are",
    explicit_rating["User-ID"].nunique(),
)

# keeps Users who have rated more than five books
books_count_df = (
    filter_df.groupby("User-ID")["ISBN"]
    .count()
    .to_frame("No-of-books-rated")
    .reset_index()
)
selected_users = list(
    books_count_df.loc[books_count_df["No-of-books-rated"] > 5, "User-ID"].unique()
)
print("Number of  users who have rated atleast 5 books are :", len(selected_users))

# dataframe with filtered number of interactions
filter_df = filter_df.loc[filter_df["User-ID"].isin(selected_users)]
print("The shape of data fame with filtered number of interactions : ", filter_df.shape)

complete_df = filter_df.copy()

complete_df["Book-Rating"].describe()


def smooth_user_preference(x):
    """Function to smooth column"""
    return math.log(1 + x, 2)


# applying function
complete_df["Book-Rating"] = complete_df["Book-Rating"].apply(smooth_user_preference)
complete_df.head()

train_df, test_df = train_test_split(
    complete_df, stratify=complete_df["User-ID"], test_size=0.20, random_state=0
)

print("# interactions on Train set: %d" % len(train_df))
print("# interactions on Test set: %d" % len(test_df))

# displaying the first 5 rows of test set
test_df.head()

# Creating a sparse pivot table with users in rows and ISBN number of books in columns
users_books_pivot_matrix_df = train_df.pivot(
    index="User-ID", columns="ISBN", values="Book-Rating"
).fillna(0)

users_books_pivot_matrix_df.head()

# Creating a matrix with the values of users_books_pivot_matrix_df
original_ratings_matrix = users_books_pivot_matrix_df.values
original_ratings_matrix[:10]

# Storing the User-IDs in a list
user_ids = list(users_books_pivot_matrix_df.index)
user_ids[:10]

# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 20

# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(original_ratings_matrix, k=NUMBER_OF_FACTORS_MF)

# converting sigma to a diagonal matrix
sigma = np.diag(sigma)

""" After the factorization, we try to to reconstruct the original matrix by multiplying its factors. The resulting matrix is not sparse any more. It has generated rating predictions for books with which users have not yet interacted (and therefore not rated), which we will use to recommend relevant books to the user."""

# Rating matric reconstructed using the matrices obtained after factorizing
predicted_ratings_matrix = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_matrix

# Converting the reconstructed matrix back to a Pandas dataframe
predicted_ratings_df = pd.DataFrame(
    predicted_ratings_matrix,
    columns=users_books_pivot_matrix_df.columns,
    index=user_ids,
).transpose()
predicted_ratings_df.head()


class CFRecommender:
    # Storing model name
    MODEL_NAME = "Collaborative Filtering"

    def __init__(self, cf_predictions_df, items_df=None):
        # Creating attributes
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        """This will return model name"""
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = (
            self.cf_predictions_df[user_id]
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={user_id: "Book-Rating"})
        )

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = (
            sorted_user_predictions[
                ~sorted_user_predictions["ISBN"].isin(items_to_ignore)
            ]
            .sort_values("Book-Rating", ascending=False)
            .head(topn)
        )

        if verbose:
            # runs only if verbose=True
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            # Merging
            recommendations_df = recommendations_df.merge(
                self.items_df, how="left", left_on="ISBN", right_on="ISBN"
            )[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]]

        return recommendations_df


# Creating object of the class
cf_recommender_model = CFRecommender(predicted_ratings_df, books_df)


def get_items_interacted(person_id, interactions_df):
    """
    This function will take user id as input and return interacted items
    """
    interacted_items = interactions_df.loc[person_id]["ISBN"]
    # Repetation is avoided by taking set
    return set(
        interacted_items if type(interacted_items) == pd.Series else [interacted_items]
    )


# Indexing by personId to speed up the searches during evaluation
full_indexed_df = complete_df.set_index("User-ID")
train_indexed_df = train_df.set_index("User-ID")
test_indexed_df = test_df.set_index("User-ID")

# Recommendation for a single user
cf_recommender_model.recommend_items(
    user_ids[3],
    items_to_ignore=get_items_interacted(user_ids[3], train_indexed_df),
    verbose=True,
)

"""##**Model Evaluation**"""


# Function for getting the set of books which a user has not interacted with
def get_not_interacted_items_sample(person_id, sample_size, seed=42):
    # Storing interacted items
    interacted_items = get_items_interacted(person_id, full_indexed_df)
    # Getting set of all items
    all_items = set(full_indexed_df["ISBN"])
    # Obtaining non interacted items
    non_interacted_items = all_items - interacted_items

    random.seed(seed)
    # Selecting random sample of given sample_size
    # non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
    non_interacted_items_list = list(non_interacted_items)
    non_interacted_items_sample = random.sample(non_interacted_items_list, sample_size)

    return set(non_interacted_items_sample)


# Top-N accuracy metrics
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            # Stores index of item id if it is present in the recommended_items
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            # If item id is not found in the recommended list
            index = -1
        # checking whether index is present in the topN items or not
        hit = int(index in range(0, topn))
        return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):

        # Getting the items in test set
        interacted_values_testset = test_indexed_df.loc[person_id]

        # Obtaining unique interacted items by the user
        if type(interacted_values_testset["ISBN"]) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset["ISBN"])
        else:
            person_interacted_items_testset = set([(interacted_values_testset["ISBN"])])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(
            person_id,
            items_to_ignore=get_items_interacted(person_id, train_indexed_df),
            topn=10000000000,
        )

        hits_at_5_count = 0
        hits_at_10_count = 0

        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:

            # Getting a random sample of 100 items the user has not interacted with
            non_interacted_items_sample = get_not_interacted_items_sample(
                person_id, sample_size=100, seed=42
            )

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[
                person_recs_df["ISBN"].isin(items_to_filter_recs)
            ]
            valid_recs = valid_recs_df["ISBN"].values

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            # Counting hit at 5
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            # Counting hit at 10
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        # Creating a dictionary
        person_metrics = {
            "hits@5_count": hits_at_5_count,
            "hits@10_count": hits_at_10_count,
            "interacted_count": interacted_items_count_testset,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
        }
        return person_metrics

    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):

        people_metrics = []

        # Calculating metrics for all users in the test set
        for idx, person_id in enumerate(list(test_indexed_df.index.unique().values)):
            # Returns dictionary containing person_metrics for each user
            person_metrics = self.evaluate_model_for_user(model, person_id)
            # Adds user_id to the dictionary
            person_metrics["_person_id"] = person_id
            # Appends each dictionary to the list
            people_metrics.append(person_metrics)

        print("%d users processed" % idx)
        # Creates dataframe containing value of metrics for all the users using the list of dictionaries
        detailed_results_df = pd.DataFrame(people_metrics).sort_values(
            "interacted_count", ascending=False
        )

        # Calculating global recall@5 and global recall@10
        global_recall_at_5 = detailed_results_df["hits@5_count"].sum() / float(
            detailed_results_df["interacted_count"].sum()
        )
        global_recall_at_10 = detailed_results_df["hits@10_count"].sum() / float(
            detailed_results_df["interacted_count"].sum()
        )

        # Creates dictionary containing global metrics
        global_metrics = {
            "modelName": model.get_model_name(),
            "recall@5": global_recall_at_5,
            "recall@10": global_recall_at_10,
        }
        return global_metrics, detailed_results_df


model_evaluator = ModelEvaluator()

print("Evaluating Collaborative Filtering (SVD Matrix Factorization) model...")
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(
    cf_recommender_model
)

print("\nGlobal metrics:\n%s" % cf_global_metrics)
cf_detailed_results_df.head(10)

import pickle

# Define the filename to save the model
model_filename = "collaborative_filtering_model.pkl"

# Save the model to a file using pickle
with open(model_filename, "wb") as file:
    pickle.dump(cf_recommender_model, file)

print(f"Model saved as {model_filename}")
