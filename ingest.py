import logging.config

logging.config.fileConfig('properties/configuration/logging.config')

loggers = logging.getLogger('ingest')


def load_files(spark,file_dir):
    try:
        loggers.warning('load_files method started...')

        df = spark.read.format('csv') \
            .option('header', True) \
            .option('inferSchema', True) \
            .load(file_dir)

    except Exception as e:
        loggers.error('An error occured at load_files method', str(e))
        raise
    else:
        loggers.warning('dataframe created successfully')

    return df


def display_df(df,df_name):
    df = df.show()

    return df


def df_count(df,df_name):
    try:
        loggers.warning('counting the records in the {}'.format(df_name))
        df_c =df.count()

    except Exception as e:
        raise
    else:
        loggers.warning('Number of records in the {} are {}'.format(df,df_c))
    return df_c