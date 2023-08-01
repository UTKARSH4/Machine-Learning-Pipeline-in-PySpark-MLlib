from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark2pmml import PMMLBuilder
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when, count, isnull
import findspark
#findspark.add_packages("org.jpmml:jpmml-sparkml:2.4.0")

conf = SparkConf().setMaster('local[*]').set("spark.jars.ivy", "C:/Users/utkarsh.verma\jr")
sc = SparkContext.getOrCreate(conf)
sqlContext = SQLContext(sc)
spark = sqlContext.sparkSession

df = spark.read.format('csv')\
    .option('header', 'True')\
    .option('inferSchema', 'True')\
    .load('C:/Users/utkarsh.verma/Downloads/archive/framingham.csv')
for col in df.columns:
    df = df.withColumn(col, df[col].cast(DoubleType()))

df.show(truncate=False)

# Check for missing values

df = df.fillna(0)
df.select([count(when(isnull(df[c]),c)).alias(c)
           for c in df.columns])\
    .show()
df = df.withColumnRenamed('TenYearCHD', 'label')
input_columns = df.drop('label', 'userId').schema.names
print("columns are:", input_columns)


def ml_model(df, input_columns):

    splits = df.randomSplit([0.8, 0.2])
    df_train = splits[0]
    df_test = splits[1]
    vector_assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

    normalizer = MinMaxScaler(inputCol="features", outputCol="features_norm")

    rf = RandomForestClassifier(featuresCol="features_norm", labelCol="label",
                                numTrees=25, featureSubsetStrategy='sqrt')
    pipeline = Pipeline(stages=[vector_assembler, normalizer, rf])

    model = pipeline.fit(df_train)
    prediction = model.transform(df_train)

    bin_eval = MulticlassClassificationEvaluator(). \
        setMetricName("accuracy"). \
        setPredictionCol("prediction"). \
        setLabelCol("label")

    print(bin_eval.evaluate(prediction))


    pmml_builder = PMMLBuilder(sc, df_train, model)
    pmml_builder.buildFile('C:/Users/utkarsh.verma/Downloads/archive/gbt.pmml')


ml_model(df,input_columns)