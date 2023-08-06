from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName('Recommender').getOrCreate()

#get sparkcontext from the sparksession
sc = spark.sparkContext
sqlContext = SQLContext(sc)

df_ratings = sqlContext.read.format('csv')\
             .option('header', True)\
             .option('inferSchema', True)\
             .load('C:/Users/utkarsh.verma/Downloads/ratings/ratings.csv')
#df_ratings.show()

# EDA
ratings = df_ratings.select('Book-Rating').toPandas()
#review_list = [reviews[i][0] for i in range(len(reviews))]
#print(ratings['Book-Rating'].tolist())
print("Unique count check", ratings['Book-Rating'].value_counts())
print("Null check", ratings['Book-Rating'].isna().sum())
plt.hist(ratings,  alpha=0.5,
         histtype='stepfilled', color='blue',
         edgecolor='none')
plt.ylabel('Frequency')
plt.xlabel('Ratings')
#plt.show()

indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ['ISBN', 'User-ID']]
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(df_ratings).transform(df_ratings)
#transformed.select(['User-ID', 'ISBN', 'User-ID_index', 'ISBN_index','Book-Rating'])

(training, test) = transformed.randomSplit([0.8, 0.2])

als_model = ALS(maxIter=5,
                regParam=0.09,
                rank=25,
                userCol="User-ID_index",
                itemCol="ISBN_index",
                ratingCol="Book-Rating",
                coldStartStrategy="drop",
                nonnegative=True)

model = als_model.fit(training)

eval = RegressionEvaluator(metricName="rmse", labelCol="Book-Rating", predictionCol="prediction")
predictions = model.transform(test)
rmse = eval.evaluate(predictions)
print("RMSE="+str(rmse))
