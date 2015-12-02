#!/bin/bash
sbt package
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/LRLBFGSWith27FeaturesPredictedLabels.txt" > LRwith27.txt
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/LRResultPredictedSolution.txt" > LRwithFeatures.txt
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/LRResultPredictedSolution1.txt" > LRwithFeatures1.txt
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/NaiveBayesResults.txt" > NaiveBayes.txt
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/RandomFOrestsResults.txt" > RandomForest.txt
/home/madhura/spark/bin/spark-submit --class "StatsCalc" --master local[*] target/scala-2.10/statscalc_2.10-1.0.jar "data/SVMResults.txt" > SVMResults.txt
cat LRwith27.txt LRwithFeatures.txt LRwithFeatures1.txt NaiveBayes.txt RandomForest.txt SVMResults.txt > FinalStatsResults.txt
