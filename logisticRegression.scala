import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

class NaiveBayes {
  def main(args: Array[String]){
  val conf = new SparkConf().setAppName("prep").setMaster("local")
    val sc = new SparkContext(conf)
 def svmlibConverter(filename:String):org.apache.spark.rdd.RDD[String] = { 
  val data = sc.textFile(filename)
  val parse=data.map { line =>{
    val strArry=line.split(',');
    var updatedLine ="";
  for(i<-0 until strArry.length){
    if(i!=2){
   updatedLine = updatedLine +strArry(i)+ " ";
    }else {
     updatedLine = updatedLine +strArry(i)+ ",";  
    }
  }
  updatedLine
  }}
val withoutHeader = parse.mapPartitionsWithIndex{(idx,iter) => if(idx == 0) iter.drop(1) else iter}

  
val newData = withoutHeader.map{line => 
      val parts = line.split(",")(0).split(" ")(2)
     (parts.toInt,line.split(",")(1).trim)}
 var vData = newData.map{case(k,v) => (k,(v.split(" ")))}

val header = parse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)

var joined = vData.map{case(k,v) => 
				var newV = for(i <- 0 until v.length)yield{
					(i+1).toString+":"+v(i).toDouble+" "
					
					}
					(k,newV.mkString.trim).toString.trim
		}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}

joined
	} 
svmlibConverter("fordTrain.csv").saveAsTextFile("trainingData")
svmlibConverter("fordTest.csv").saveAsTextFile("testingData")


val traindata = MLUtils.loadLibSVMFile(sc, "trainingData/part-0000*").cache()
val testdata = MLUtils.loadLibSVMFile(sc, "testingData/part-0000*")
	

val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(traindata)

// Compute raw scores on the test set.
val predictionAndLabels = testdata.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
predictionAndLabels.coalesce(1).saveAsTextFile("LRResultPredictedSolution")
}
}
