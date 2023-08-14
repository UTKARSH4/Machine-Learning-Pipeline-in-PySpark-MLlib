from pyspark.sql import SparkSession
import logging.config

logging.config.fileConfig('properties/configuration/logging.config')

loggers = logging.getLogger('create_spark')


def create_spark_object():
    try:
        loggers.info('get_spark_object method started')
        spark = SparkSession.builder.master("local").appName("word").getOrCreate()
        return spark

    except Exception as e:
        loggers.error('Error occured in the ger_spark_object', str(e))
        raise

    else:
        loggers.info('Spark object created')