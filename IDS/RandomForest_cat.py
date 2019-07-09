from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.shell import sqlContext
import time

spark_yog = SparkSession.builder.appName('project').getOrCreate()
df_yog = spark_yog.read.csv('Data_Combined.csv', header = True, inferSchema = True)
# df_yog.printSchema()


# processing data

cloum_set=df_yog.columns
categorical_Columns = ['proto', 'service', 'state']
stages_feat = []

# changes the data into vectors that library can undestand and saved in stages_feat

for categorical_Col in categorical_Columns:
    stringIndexer = StringIndexer(inputCol = categorical_Col, outputCol = categorical_Col + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categorical_Col + "classVec"])
    stages_feat += [stringIndexer, encoder]

# Change/Feed Label
Label_str_Indexer = StringIndexer(inputCol = 'attack_cat', outputCol = 'Pred_Label')
stages_feat += [Label_str_Indexer]

#all labels used
numeric_Cols=['dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl'	,'ct_dst_ltm',	'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']
# Random 5 labels
# numeric_Cols= ['sttl', 'sbytes', 'dwin', 'smean', 'dmean']
# Another set of random 5 labels
# numeric_Cols = ['sloss', 'sload', 'stcpb', 'dtcpb', 'trans_depth']
assembler_Inputs = [c + "classVec" for c in categorical_Columns] + numeric_Cols
#labeling the output
Assembler = VectorAssembler(inputCols=assembler_Inputs, outputCol="features_all")
stages_feat += [Assembler]

# Pipeline processing of data to take into the library
pipeline = Pipeline(stages = stages_feat)
pipelineModel = pipeline.fit(df_yog)
df_yog = pipelineModel.transform(df_yog)
selected_Cols = ['Pred_Label', 'features_all'] + cloum_set
df_yog = df_yog.select(selected_Cols)
# df_yog.printSchema()

# splits = df_yog.randomSplit([0.6,0.4], 1234)
training_data, testing_data = df_yog.randomSplit([0.6805,0.3195], seed = 99999999)
print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(testing_data.count()))

#Model
RandomForest = RandomForestClassifier(labelCol="Pred_Label", featuresCol="features_all", numTrees=10)

start=time.time()
RandomForestModel = RandomForest.fit(training_data)
end=time.time()
start1=time.time()
f_predictions = RandomForestModel.transform(testing_data)
end1=time.time()


# #PRINT CONFUSION MATRIX
# Cm=f_predictions.select("PoutLabel","label").distinct().toPandas()
# f_predictions.groupBy("PoutLabel","prediction").count().show()


print("Time to train:")
print(end-start)
print("Time to predict:")
print(end1-start1)

evaluu = BinaryClassificationEvaluator(labelCol="Pred_Label",rawPredictionCol="prediction")
print("Accuracy of model")
print(evaluu.evaluate(f_predictions))
# print(evaluu.explainParams())

# #MULT CLASS
evaluator =MulticlassClassificationEvaluator(labelCol="Pred_Label",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(f_predictions)
print("ACCURACY :")
print(accuracy)

evaluator =MulticlassClassificationEvaluator(labelCol="Pred_Label",predictionCol="prediction", metricName="f1")
score= evaluator.evaluate(f_predictions)
print("F1-score")
print(score)


