# Copyright 2017 onwards, fast.ai, Inc.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import datetime
import os
import time
import pandas as pd
from distutils.version import LooseVersion

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import SparkConf, Row
from pyspark.sql import SparkSession


if __name__ == '__main__':

    # ================ #
    # DATA PREPARATION #
    # ================ #

    print('================')
    print('Data preparation')
    print('================')

    # Create Spark session for data preparation.
    conf = SparkConf().setAppName('Keras Spark Rossmann Estimator Example').set('spark.sql.shuffle.partitions', '6')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    data_dir = "hdfs:///user/tgraves/rossmann/"
    train_csv = spark.read.csv('%s/train.csv' % data_dir, header=True)
    test_csv = spark.read.csv('%s/test.csv' % data_dir, header=True)

    store_csv = spark.read.csv('%s/store.csv' % data_dir, header=True)
    store_states_csv = spark.read.csv('%s/store_states.csv' % data_dir, header=True)
    state_names_csv = spark.read.csv('%s/state_names.csv' % data_dir, header=True)
    google_trend_csv = spark.read.csv('%s/googletrend.csv' % data_dir, header=True)
    weather_csv = spark.read.csv('%s/weather.csv' % data_dir, header=True)


    def expand_date(df):
        print("Tom print schema expand date")
        df.printSchema()
        df = df.withColumn('Date', df.Date.cast(T.DateType()))
        return df.withColumn('Year', F.year(df.Date)) \
            .withColumn('Month', F.month(df.Date)) \
            .withColumn('Week', F.weekofyear(df.Date)) \
            .withColumn('Day', F.dayofmonth(df.Date))
#
       #return df \
        #    .withColumn("Week", ((df.Month * 4) + df.Day / 7).cast(T.IntegerType()))

    #def expand_date(df):
    #    df = df.withColumn('Date', df.Date.cast(T.DateType()))
    #    return df.withColumn('Year', F.year(df.Date)) \
    #        .withColumn('Month', F.month(df.Date)) \
    #        .withColumn("Week", (((F.month(df.Date) * 4) + F.dayofmonth(df.Date) / 7) - 2).cast(T.IntegerType())) \
    #        .withColumn('Day', F.dayofmonth(df.Date))


    def prepare_google_trend():
        # Extract week start date and state.
        google_trend_all = google_trend_csv \
            .withColumn('Date', F.substring(google_trend_csv.week, 0, 10)) \
            .withColumn('State', F.substring(google_trend_csv.file, 13, 2))
        # Map state NI -> HB,NI to align with other data sources.
        google_trend_all = google_trend_all \
            .withColumn('State', F.when(google_trend_all.State == 'NI', 'HB,NI').otherwise(google_trend_all.State))
        # Expand dates.
        return expand_date(google_trend_all)


    def add_elapsed_udf(df, cols):
        for col in cols:
            df.createOrReplaceTempView("horovoddesc")
            windowDesc = """
                SELECT *, nvl(max_dt_with_promo_true, min_dt) effective_promo_date FROM (
                  SELECT *,
                         min(Date) over (partition by Store order by Date desc rows between unbounded preceding and current row) as min_dt,
                         max(case when Promo then Date end) over (partition by Store order by Date desc rows between unbounded preceding and current row) as max_dt_with_promo_true
                  FROM horovoddesc
                  ORDER BY Store ASC, Date ASC )
            """
            df = spark.sql(windowDesc)
            df = df.withColumn('Before%s' % col, F.datediff(df.Date, df.effective_promo_date).cast(T.LongType())).drop('max_dt_with_promo_true').drop('effective_promo_date').drop('min_dt')
        for col in cols:
            df.createOrReplaceTempView("horovod")
            window = """
                SELECT *, nvl(max_dt_with_promo_true, min_dt) effective_promo_date FROM (
                  SELECT *,
                         min(Date) over (partition by Store order by Date asc rows between unbounded preceding and current row) as min_dt,
                         max(case when Promo then Date end) over (partition by Store order by Date asc rows between unbounded preceding and current row) as max_dt_with_promo_true
                  FROM horovod
                  ORDER BY Store ASC, Date ASC )
            """
            df = spark.sql(window)
            df = df.withColumn('After%s' % col, F.datediff(df.Date, df.effective_promo_date).cast(T.LongType())).drop('max_dt_with_promo_true').drop('effective_promo_date').drop('min_dt')
        return df


    def add_elapsed(df, cols):
        def add_elapsed_column(col, asc):
            def fn(rows):
                last_store, last_date = None, None
                for r in rows:
                    if last_store != r.Store:
                        last_store = r.Store
                        last_date = r.Date
                    if r[col]:
                        last_date = r.Date
                    fields = r.asDict().copy()
                    fields[('After' if asc else 'Before') + col] = (r.Date - last_date).days
                    yield Row(**fields)
            return fn

        df = df.repartition(df.Store)
        for asc in [False, True]:
            sort_col = df.Date.asc() if asc else df.Date.desc()
            rdd = df.sortWithinPartitions(df.Store.asc(), sort_col).rdd
            for col in cols:
                rdd = rdd.mapPartitions(add_elapsed_column(col, asc))
            df = rdd.toDF()
        return df


    def prepare_df(df):
        num_rows = df.count()

        # Expand dates.
        df.printSchema()
        df = expand_date(df)

        df = df \
            .withColumn('Open', df.Open != '0') \
            .withColumn('Promo', df.Promo != '0') \
            .withColumn('StateHoliday', df.StateHoliday != '0') \
            .withColumn('SchoolHoliday', df.SchoolHoliday != '0')

        # Merge in store information.
        store = store_csv.join(store_states_csv, 'Store')
        df = df.join(store, 'Store')

        # Merge in Google Trend information.
        google_trend_all = prepare_google_trend()
        df = df.join(google_trend_all, ['State', 'Year', 'Week']).select(df['*'], google_trend_all.trend)

        # Merge in Google Trend for whole Germany.
        google_trend_de = google_trend_all[google_trend_all.file == 'Rossmann_DE']
        google_trend_de = google_trend_de.withColumnRenamed('trend', 'trend_de')
        df = df.join(google_trend_de, ['Year', 'Week']).select(df['*'], google_trend_de.trend_de)

        # Merge in weather.
        weather = weather_csv.join(state_names_csv, weather_csv.file == state_names_csv.StateName)
        df = df.join(weather, ['State', 'Date'])

        # Fix null values.
        df = df \
            .withColumn('CompetitionOpenSinceYear', F.coalesce(df.CompetitionOpenSinceYear, F.lit(1900))) \
            .withColumn('CompetitionOpenSinceMonth', F.coalesce(df.CompetitionOpenSinceMonth, F.lit(1))) \
            .withColumn('Promo2SinceYear', F.coalesce(df.Promo2SinceYear, F.lit(1900))) \
            .withColumn('Promo2SinceWeek', F.coalesce(df.Promo2SinceWeek, F.lit(1)))

        # Days & months competition was open, cap to 2 years.
        df = df.withColumn('CompetitionOpenSince',
                           F.to_date(F.concat(df.CompetitionOpenSinceYear, F.lit('-'), df.CompetitionOpenSinceMonth, F.lit('-15'))))

        #df = df.withColumn('CompetitionOpenSince',
        #                   F.to_date(F.format_string('%s-%s-15', df.CompetitionOpenSinceYear,
        #                                             df.CompetitionOpenSinceMonth)))

        df = df.withColumn('CompetitionDaysOpen',
                           F.when(df.CompetitionOpenSinceYear > 1900,
                                  F.greatest(F.lit(0), F.least(F.lit(360 * 2), F.datediff(df.Date, df.CompetitionOpenSince))))
                           .otherwise(0))
        df = df.withColumn('CompetitionMonthsOpen', (df.CompetitionDaysOpen / 30).cast(T.IntegerType()))

        # Days & weeks of promotion, cap to 25 weeks.
        #df = df.withColumn('Promo2Since',
        #                   F.date_add(F.concat(df.Promo2SinceYear, F.lit('-01-01')), (df.Promo2SinceWeek.cast(T.IntegerType()) - 1) * 7))

        df = df.withColumn('Promo2Since',
                           F.expr('date_add(concat(Promo2SinceYear, "-01-01"), (cast(Promo2SinceWeek as int) - 1) * 7)'))

        #df = df.withColumn('Promo2Since',
        #                   F.expr('date_add(format_string("%s-01-01", Promo2SinceYear), (cast(Promo2SinceWeek as int) - 1) * 7)'))

        df = df.withColumn('Promo2Days',
                           F.when(df.Promo2SinceYear > 1900,
                                  F.greatest(F.lit(0), F.least(F.lit(25 * 7), F.datediff(df.Date, df.Promo2Since))))
                           .otherwise(0))
        df = df.withColumn('Promo2Weeks', (df.Promo2Days / 7).cast(T.IntegerType()))

        # Check that we did not lose any rows through inner joins.
        #assert num_rows == df.count(), 'lost rows in joins'
        return df



    def cast_columns(df, cols):
        for col in cols:
            df = df.withColumn(col, F.coalesce(df[col].cast(T.FloatType()), F.lit(0.0)))
        return df


    # Prepare data frames from CSV files.
    train_df = prepare_df(train_csv).cache()
    test_df = prepare_df(test_csv).cache()

    # Add elapsed times from holidays & promos, the data spanning training & test datasets.
    elapsed_cols = ['Promo', 'StateHoliday', 'SchoolHoliday']
    #elapsed = add_elapsed(train_df.select('Date', 'Store', *elapsed_cols)
    #                      .unionAll(test_df.select('Date', 'Store', *elapsed_cols)),
    #                      elapsed_cols)
#
    elapsed = add_elapsed_udf(train_df.select('Date', 'Store', *elapsed_cols)
                      .unionAll(test_df.select('Date', 'Store', *elapsed_cols)),
                      elapsed_cols)

    #print('Tom schema')
    #elapsed_new.printSchema()
    #elapsed.printSchema()
    #elapsed.show()

    # Join with elapsed times.
    train_df = train_df \
        .join(elapsed, ['Date', 'Store']) \
        .select(train_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])
    test_df = test_df \
        .join(elapsed, ['Date', 'Store']) \
        .select(test_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])

    # Filter out zero sales.
    train_df = train_df.filter(train_df.Sales > 0)

    print('===================')
    print('Prepared data frame')
    print('===================')
    train_df.show()

    categorical_cols = [
        'Store', 'State', 'DayOfWeek', 'Year', 'Month', 'Day', 'Week', 'CompetitionMonthsOpen', 'Promo2Weeks', 'StoreType',
        'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'Events', 'Promo',
        'StateHoliday', 'SchoolHoliday'
    ]

    continuous_cols = [
        'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity',
        'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
        'BeforePromo', 'AfterPromo', 'AfterStateHoliday', 'BeforeStateHoliday', 'BeforeSchoolHoliday', 'AfterSchoolHoliday'
    ]

    all_cols = categorical_cols + continuous_cols

    # Select features.
    train_df = train_df.select(*(all_cols + ['Sales', 'Date'])).cache()
    test_df = test_df.select(*(all_cols + ['Id', 'Date'])).cache()

    #unioned_df = train_df.select(*categorical_cols).unionAll(test_df.select(*categorical_cols))

    def build_vocabulary(df, cols):
        vocab = {}
        for col in cols:
            values = [r[0] for r in df.select(col).distinct().collect()]
            col_type = type([x for x in values if x is not None][0])
            default_value = col_type()
            vocab[col] = sorted(values, key=lambda x: x or default_value)
        return vocab

    vocab = build_vocabulary(train_df.select(*categorical_cols)
                             .unionAll(test_df.select(*categorical_cols)), #.cache(),
                             categorical_cols)

    def lookup_columns(df, vocab):
        def lookup(mapping):
            def fn(v):
                return mapping.index(v)
            return F.udf(fn, returnType=T.IntegerType())

        for col, mapping in vocab.items():
            df = df.withColumn(col, lookup(mapping)(df[col]))
        return df


    #cat_dfs = {}
    #for col in categorical_cols:
    #    stateDf = unioned_df.select(col).distinct().sort(col)
    #    window = Window.orderBy(F.col(col))
    #    df_final = stateDf.withColumn('row_number', F.row_number().over(window)) # .withColumnRenamed(col, '%sToJoin' % col)
    #    cat_dfs[col] = df_final

   # broadcast_dfs = {}
   # for col in categorical_cols:
   #     df = pd.DataFrame(vocab[col])
   #     df['index_col'] = df.index
   #     df.columns = [col, 'index%s' % col]
   #     ddf = spark.createDataFrame(df)
   #     broadcast_dfs[col] = ddf
#
#    for col in categorical_cols:
#        ddf = broadcast_dfs[col]
#        cat_indexed = train_df.join(F.broadcast(ddf), [col], 'left_outer')
##        train_df = cat_indexed.drop(col).withColumnRenamed('index%s' % col, col)


#    train_df = cast_columns(train_df, continuous_cols + ['Sales'])  

    #test_cat_dfs = {}
    #print("starting window")
    #for col in categorical_cols:
    #    stateDf = unioned_df.select(col).distinct().sort(col)
    #    window = Window.orderBy(F.col(col))
    #    df_final = stateDf.withColumn('row_number', F.row_number().over(window)) #.withColumnRenamed(col, '%sToJoin' % col)
    #    test_cat_dfs[col] = df_final
    #print("ending window")

#    for col in categorical_cols:
#        ddf = broadcast_dfs[col]
#        cat_indexed = test_df.join(F.broadcast(ddf), [col], 'left_outer')
#        test_df = cat_indexed.drop(col).withColumnRenamed('index%s' % col, col)


    #for col in test_cat_dfs:
    #    cat_df = test_cat_dfs[col]
    #    #joinCol = '%sToJoin' % col
    #    #cat_indexed = test_df.join(cat_df.hint("broadcast"), train_df[col] == cat_df[joinCol], 'left_outer').drop(joinCol)
    #    cat_indexed = test_df.join(cat_df.hint("broadcast"), [col], 'left_outer') #.drop(joinCol)
    #    test_df = cat_indexed.drop(col).withColumnRenamed('row_number', col)

#    test_df = cast_columns(test_df, continuous_cols)

    train_df = cast_columns(train_df, continuous_cols + ['Sales'])
    train_df = lookup_columns(train_df, vocab)
    test_df = cast_columns(test_df, continuous_cols)
    test_df = lookup_columns(test_df, vocab)


    # Split into training & validation.
    # Test set is in 2015, use the same period in 2014 from the training set as a validation set.
    test_min_date = test_df.agg(F.min(test_df.Date)).collect()[0][0]
    test_max_date = test_df.agg(F.max(test_df.Date)).collect()[0][0]
    one_year = datetime.timedelta(365)
    train_df = train_df.withColumn('Validation',
                                   (train_df.Date > test_min_date - one_year) & (train_df.Date <= test_max_date - one_year))

    # Determine max Sales number.
    max_sales = train_df.agg(F.max(train_df.Sales)).collect()[0][0]

    #print('tom schema 1')
    #train_df.printSchema()

    # Convert Sales to log domain
    train_df = train_df.withColumn('Sales', F.log(train_df.Sales))

    #print('tom schema 1')
    #train_df.printSchema()

    #print('===================================')
    #print('Data frame with transformed columns')
    #print('===================================')

    print('================')
    print('Data frame sizes')
    print('================')
    train_rows = train_df.filter(~train_df.Validation).count()
    val_rows = train_df.filter(train_df.Validation).count()
    test_rows = test_df.count()
    print('Training: %d' % train_rows)
    print('Validation: %d' % val_rows)
    print('Test: %d' % test_rows)
#
    train_df.select(categorical_cols + continuous_cols + ['Sales', 'Date', 'Validation']).write.mode("overwrite").parquet("hdfs:///user/tgraves/horovodtrainoutnoudf")
    test_df.select(categorical_cols + continuous_cols + ['Id']).write.mode("overwrite").parquet("hdfs:///user/tgraves/horovodtestoutnoudf")

    spark.stop()
