from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
import time
from pyspark.shell import sqlContext

spark_yog = SparkSession.builder.appName('project_CSE15055').getOrCreate()
df_yog = spark_yog.read.csv('Data_Combined.csv', header = True, inferSchema = True)
# df_yog.printSchema()


# processing data

column_set=df_yog.columns
categorical_Columns = ['proto', 'service', 'state']
stages_feat = []

# changes the data into vectors that library can undestand and saved in stages_feat

for categorical_Col in categorical_Columns:
    string_Indexer = StringIndexer(inputCol = categorical_Col, outputCol = categorical_Col + 'Index')
    encoder_cat_col = OneHotEncoderEstimator(inputCols=[string_Indexer.getOutputCol()], outputCols=[categorical_Col + "classVec"])
    stages_feat += [string_Indexer, encoder_cat_col]

# Change/Feed Label
Label_str_Indexer = StringIndexer(inputCol = 'attack_cat', outputCol = 'Pred_Label')
stages_feat += [Label_str_Indexer]

#all labels used
numeric_cols=['dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl'	,'ct_dst_ltm',	'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']
# Random 5 labels
# numeric_cols= ['sttl', 'sbytes', 'dwin', 'smean', 'dmean']
# Another set of random 5 labels
# numeric_cols = ['sloss', 'sload', 'stcpb', 'dtcpb', 'trans_depth']
assembler_Inputs = [c + "classVec" for c in categorical_Columns] + numeric_cols
#labeling the output
Assembler = VectorAssembler(inputCols=assembler_Inputs, outputCol="features_all")
stages_feat += [Assembler]

# Pipeline processing of data to take into the library
pipeline = Pipeline(stages = stages_feat)
pipelineModel = pipeline.fit(df_yog)
df_yog = pipelineModel.transform(df_yog)
selected_cols = ['Pred_Label', 'features_all'] + column_set
df_yog = df_yog.select(selected_cols)
# df_yog.printSchema()


# splits = df_yog.randomSplit([0.6,0.4], 1234)
training_data, testing_data = df_yog.randomSplit([0.6805,0.3195], seed = 99999999)
print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(testing_data.count()))

#Model
NaiveBayes = NaiveBayes(labelCol="Pred_Label", featuresCol="features_all", smoothing=1.0, modelType="multinomial" )
# print("aa")
start=time.time()
NaiveBayesModel = NaiveBayes.fit(training_data)
end=time.time()
start1=time.time()
f_predictions = NaiveBayesModel.transform(testing_data)
end1=time.time()
# f_predictions.select('Pred_Label','prediction').show(85000)
# f_predictions.select('Pred_Label','prediction').coalesce(1).write.option("header","true").csv('ss.csv')
# #PRINT CONFUSION MATRIX
# Cm=f_predictions.select("PoutLabel","label").distinct().toPandas()
# f_predictions.groupBy("PoutLabel","prediction").count().show()

# f_predictions.select('Pred_Label','prediction','label','dur').coalesce(1).write.option("header","true").csv('NaivesBayesPredictionResult_cat.csv')

print("Time of train:")
print(end-start)
print("time of predict:")
print(end1-start1)
# f_predictions.show()


evaluu = BinaryClassificationEvaluator(labelCol="Pred_Label",rawPredictionCol="prediction")
print("Accuracy of model")
print(evaluu.evaluate(f_predictions))
# print(evaluu.explainParams())

evaluator =MulticlassClassificationEvaluator(labelCol="Pred_Label",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(f_predictions)
print("ACCURACY :")
print(accuracy)

evaluator =MulticlassClassificationEvaluator(labelCol="Pred_Label",predictionCol="prediction", metricName="f1")
score = evaluator.evaluate(f_predictions)
print("F1-score :")
print(score)
