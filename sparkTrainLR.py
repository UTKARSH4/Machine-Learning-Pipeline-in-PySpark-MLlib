from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark2pmml import PMMLBuilder
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when, count, isnull
import logging
import sys
#import findspark
#findspark.add_packages("org.jpmml:jpmml-sparkml:2.4.0")


from create_spark import create_spark_object
from validate import get_current_date
from ingest import load_files, display_df, df_count

logging.config.fileConfig('properties/configuration/logging.config')


def main():
    try:
        logging.info('I am the main method')
        logging.info('calling spark object')
        spark = create_spark_object()

        #logging.info('object create...', str(spark))
        logging.info('validating spark object')
        get_current_date(spark)

        logging.info('reading file....')
        df = load_files(spark, file_dir='/home/utkarsh/Downloads/archive/framingham.csv')
        logging.info('displaying the dataframe... ')
        display_df(df,'df')
        logging.info('validating the dataframe... ')
        df_count(df,'df')

        for col in df.columns:
            df = df.withColumn(col, df[col].cast(DoubleType()))

        df.show(truncate=False)

        # Check for missing values

        df = df.fillna(0)
        df.select([count(when(isnull(df[c]),c)).alias(c)
                   for c in df.columns]).show()
        df = df.withColumnRenamed('TenYearCHD', 'label')
        input_columns = df.drop('label', 'userId').schema.names
        print("columns are:", input_columns)
        ml_model(spark,df, input_columns)

    except Exception as e:
        logging.error('An error occured ===', str(e))
        sys.exit(1)


def ml_model(spark,df, input_columns):

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

    pmml_builder = PMMLBuilder(spark, df_train, model)
    pmml_builder.buildFile('/home/utkarsh/Downloads/archive/gbt.pmml')


if __name__ == '__main__':
    main()
    logging.info('Application done')

