import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object SVM {
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("prep").setMaster("local")
    val sc = new SparkContext(conf)
  
  val traindata = sc.textFile("fordTrain.md")
  val parse=traindata.map { line =>{
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
 var vData = newData.map{case(k,v) => (k,{
   //v.split(" ")
   val s= v(1).split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  var prevVal=0.0
  while(i<s.length){
    var value=0.0
    if(i%30!=7 || i%30!=25 || i%30!=27){
    value=s(i).toDouble
    
    
    list+=value
    }
    
    i+=1
  }
   list.toArray
   })}

val header = parse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

//val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)
var joined = vData.map{case(k,v) => var newV = for(i <- 0 until v.length)yield{(i+1).toString+":"+v(i).toDouble+" "}
(k,newV.mkString.trim).toString.trim}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}
joined.saveAsTextFile("TrainResult")

/////////////////////////////test data preprocessing
val testdata = sc.textFile("fordTest.md")
  val testparse=testdata.map { line =>{
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
    val testwithoutHeader = testparse.mapPartitionsWithIndex{(idx,iter) => if(idx == 0) iter.drop(1) else iter}

  
val newTestData = testwithoutHeader.map{line => 
      val parts = line.split(",")(0).split(" ")(2)
     (parts.toInt,line.split(",")(1).trim)}
 var vTestData = newTestData.map{case(k,v) => (k,{
   //v.split(" ")
   val s= v(1).split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  var prevVal=0.0
  while(i<s.length){
    var value=0.0
    if(i%30!=7 || i%30!=25 || i%30!=27){
    value=s(i).toDouble
    
    
    list+=value
    }
    
    i+=1
  }
   list.toArray
 })}
 }

val testheader = testparse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

//val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)
var testJoined = vTestData.map{case(k,v) => var newV = for(i <- 0 until v.length)yield{(i+1).toString+":"+v(i).toDouble+" "}
(k,newV.mkString.trim).toString.trim}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}
testJoined.saveAsTextFile("TestResult")
///////////////////////////////////////////////////////////////////////////

// Load training data in LIBSVM format.
val TrainData = MLUtils.loadLibSVMFile(sc, "TrainResult/part-0000*")
val TestData = MLUtils.loadLibSVMFile(sc, "TestResult/part-0000*")

/*// Split data into training (60%) and test (40%).
val splits = TrainData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)*/

/*val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(TrainData)
val predictionAndLabels = TestData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)*/

val numIterations = 100
val model = SVMWithSGD.train(TrainData, numIterations)

// Clear the default threshold.
//model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = TestData.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC = " + auROC)

val metrics1 = new MulticlassMetrics(scoreAndLabels)
val precision = metrics1.precision
println("Precision = " + precision)

  }
  
}