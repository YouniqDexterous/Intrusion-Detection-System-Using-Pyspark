from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.shell import sqlContext
from pyspark.mllib.util import MLUtils
import time
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession, SQLContext
#data input

spark_yog = SparkSession.builder.appName('project_CSE15055').getOrCreate()
df_yog = spark_yog.read.csv('Data_Combined.csv', header = True, inferSchema = True)
df_yog.printSchema()

# processing data

cloumns_set=df_yog.columns
categorical_Columns = ['proto',  'state', 'service']
stages_feat = []

# changes the data into vectors that library can undestand and saved in stages_feat
# df_yog.show()
for categorical_Col in categorical_Columns:
    string_Indexer = StringIndexer(inputCol = categorical_Col, outputCol = categorical_Col + 'Index')
    encoder_cat_col = OneHotEncoderEstimator(inputCols=[string_Indexer.getOutputCol()], outputCols=[categorical_Col + "classVec"])
    stages_feat += [string_Indexer, encoder_cat_col]

# Change/Feed Label
Label_str_Indexer = StringIndexer(inputCol = 'label', outputCol = 'Pred_Label')
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
# df_yog.show(3)
# Pipeline processing of data to take into the library
pipeline = Pipeline(stages = stages_feat)
pipelineModel = pipeline.fit(df_yog)
df_yog = pipelineModel.transform(df_yog)



selected_Cols = ['Pred_Label', 'features_all'] + cloumns_set

df_yog = df_yog.select(selected_Cols)
# df_yog.show(3)
# df_yog.show()
# split data
# split used in paper training_data= 175341(0.6805) testing_data=82332(0.3195) total: 257673
training_data, testing_data = df_yog.randomSplit([0.6805,0.3195], seed = 2017)


print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(testing_data.count()))
# testing_data.show(3)

# model buit and evaluation
# ----------------    Description about the evaluation elements ---------------------------------
# input of Decision Model   -->label (label to predict labelcol) { to access use labelCol }
#                           -->features_all (feature use for prediction in vector form) { to access use features_allCol }
#output of Decision --> prediction(actual predicton of model) {to access use predictionCol}
#                   --> rawPrediction()                        { to access use rawPredictionCol }
#                   --> probability()                           { to access ues probabilityCol }



# Train the model
DecisonM = DecisionTreeClassifier(featuresCol='features_all', labelCol='Pred_Label', maxDepth=20)

start=time.time()
DecisionModel = DecisonM.fit(training_data)
end=time.time()
start1=time.time()
f_predictions = DecisionModel.transform(testing_data)
end1=time.time()


# #PRINT CONFUSION MATRIX
Cm=f_predictions.select("Pred_Label","label").distinct().toPandas()


f_predictions.groupBy("Pred_Label","prediction").count().show()


print("Time to train:")
print(end-start)
print("Time to predict:")
print(end1-start1)

print("\n \t\t\tSCHEMA AFTER PREDICTION \n")
f_predictions.select('label','prediction', 'Pred_Label').show()

# predictionAndLabels = testing_data.map(lambda lp: (float(DecisonM.predict(lp.features_all)), lp.label))
# Print the f_predictions in table format
# f_predictions.select('rawPrediction','probability','prediction', 'Pred_Label','label').show()
# f_predictions.select('Pred_Label','prediction').show()

# f_predictions.select('Pred_Label','prediction').coalesce(1).write.option("header","true").csv('decision.csv')
#To print full data set
# for row in f_predictions.collect():
#     print(row)
# print(DecisonM.explainParams())

#EVALUATION
evaluu = BinaryClassificationEvaluator(labelCol="Pred_Label",rawPredictionCol="prediction")
print("Accuracy of model")
print(evaluu.evaluate(f_predictions))
# print(evaluu.explainParams())


# #MULTICLASS
evaluator = MulticlassClassificationEvaluator(labelCol="Pred_Label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(f_predictions)
print("ACCURACY :")
print(accuracy)

evaluator = MulticlassClassificationEvaluator(labelCol="Pred_Label", predictionCol="prediction",metricName="f1")
f1accuracy = evaluator.evaluate(f_predictions)
print("F1-score :")
print(f1accuracy)

