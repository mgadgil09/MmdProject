import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest

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
    if(i==2){
   updatedLine = updatedLine +strArry(i)+ ",";
    }
    else if(i==10 || i==28 || i==30){
	updatedLine = updatedLine +strArry(i)+ "r ";
	}
	else {
     updatedLine = updatedLine +strArry(i)+ " ";  
    }
  }
  updatedLine.trim
  }}
val parsedData = parse.map{line => (line.split(",")(0).split(" ")(2),line.split(",")(1))}

val withoutHeader = parsedData.mapPartitionsWithIndex{(idx,iter) => if(idx == 0) iter.drop(1) else iter}

  

val impData = withoutHeader.map{case(k,arr) =>
     var array = arr.split(" ")
     var temp = for(i <- 0 to array.length-1 if (!array(i).endsWith("r")))yield{
		array(i)+" "
			}
	(k,temp.toArray.mkString.trim)
	  }
 var vData = impData.map{case(k,v) => (k,(v.split(" ")))}

val header = parse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)

var joined = vData.map{case(k,v) => 
				var newV = for(i <- 0 until v.length)yield{
					(i+1).toString+":"+v(i).toDouble+" "
					
					}
					(k,newV.mkString.trim).toString.trim
		}.map{line => line.substring(1,line.length-1)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}

joined
	} 	
svmlibConverter("fordTrain.csv").saveAsTextFile("trainingDataRandomForests")
svmlibConverter("fordTest.csv").saveAsTextFile("testingDataRandomForests")


val traindata = MLUtils.loadLibSVMFile(sc, "trainingDataRandomForests/part-0000*").cache()
val testdata = MLUtils.loadLibSVMFile(sc, "testingDataRandomForests/part-0000*")

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 27// Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 2
val maxBins = 32

val model = RandomForest.trainClassifier(traindata, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testdata.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val accuracy = labelAndPreds.filter(r => r._1==r._2).count.toDouble / labelAndPreds.count()
val metrics = new BinaryClassificationMetrics(labelAndPreds)
var m = metrics.roc().collect()