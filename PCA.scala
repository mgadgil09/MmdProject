import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel,  LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
 
def svmlibConverterForPCA(trainDataFilename:String, testDataFilename:String):(org.apache.spark.rdd.RDD[String],org.apache.spark.rdd.RDD[String]) = {
 val traindata = sc.textFile(trainDataFilename)
 val testdata = sc.textFile(testDataFilename)
 val trainparse=traindata.map { line =>{
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
val trainwithoutHeader = trainparse.mapPartitionsWithIndex{(idx,iter) => if(idx == 0) iter.drop(1) else iter}
val testwithoutHeader = testparse.mapPartitionsWithIndex{(idx,iter) => if(idx == 0) iter.drop(1) else iter}
val trainIsAlertCol = trainwithoutHeader.map{line => line.split(",")(0)}.map{line => line.split(" ")(2)}.zipWithIndex.map{case(k,v) => (v,k)}
val testIsAlertCol = testwithoutHeader.map{line => line.split(",")(0)}.map{line => line.split(" ")(2)}.zipWithIndex.map{case(k,v) => (v,k)}
var trainvectorValue = trainwithoutHeader.map{line => line.split(",")(1)}.map{line => 
      var doub = line.split(" ")
      var doubVal = for(i <- 0 until doub.length)yield{
      doub.apply(i).toDouble
      }
      Vectors.dense(doubVal.toArray)}
var testvectorValue = testwithoutHeader.map{line => line.split(",")(1)}.map{line => 
      var doub = line.split(" ")
      var doubVal = for(i <- 0 until doub.length)yield{
      doub.apply(i).toDouble
      }
      Vectors.dense(doubVal.toArray)}
val trainPCAmat : RowMatrix = new RowMatrix(trainvectorValue)
val testPCAmat : RowMatrix = new RowMatrix(testvectorValue)
val top20pc = trainPCAmat.computePrincipalComponents(100)
val train20 = trainPCAmat.multiply(top20pc).rows.zipWithIndex.map{case(k,v) => (v,k)}
val test20 = testPCAmat.multiply(top20pc).rows.zipWithIndex.map{case(k,v) => (v,k)}

val trainJoined = trainIsAlertCol.join(train20).map{case(a,(b,c)) => (b,c)}
val testJoined = testIsAlertCol.join(test20).map{case(a,(b,c)) => (b,c)}
var trainDataFinal = trainJoined.map{case(k,v) => 
				var newV = for(i <- 0 until v.toArray.length)yield{
					(i+1).toString+":"+v(i).toDouble+" "
					
					}
					(k.toInt,newV.mkString.trim).toString.trim
		}.map{line => line.substring(1,line.length-1)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}
var testDataFinal = testJoined.map{case(k,v) => 
				var newV = for(i <- 0 until v.toArray.length)yield{
					(i+1).toString+":"+v(i).toDouble+" "
					
					}
					(k.toInt,newV.mkString.trim).toString.trim
		}.map{line => line.substring(1,line.length-1)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}


(trainDataFinal, testDataFinal)
}

		val trainAndTestData = svmlibConverterForPCA("fordTrain.csv", "fordTest.csv")//.saveAsTextFile("trainingDataForPCA")
		//svmlibConverterForPCA("fordTest.csv").saveAsTextFile("testingDataForPCA")
		trainAndTestData._1.saveAsTextFile("trainingDataForPCA100")
		trainAndTestData._2.saveAsTextFile("testingDataForPCA100")

	val traindata = MLUtils.loadLibSVMFile(sc, "trainingDataForPCA100/part-0000*").cache()
	val testdata = MLUtils.loadLibSVMFile(sc, "testingDataForPCA100/part-0000*")
	

	val model = new  LogisticRegressionWithLBFGS().setNumClasses(2).run(traindata)

	// Compute raw scores on the test set.
	val predictionAndLabels = testdata.map { case LabeledPoint(label, features) =>
	  val prediction = model.predict(features)
	  (prediction, label)
	}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
predictionAndLabels.filter{x=>x._1==x._2}.count().toDouble/ predictionAndLabels.count()
predictionAndLabels.coalesce(1).saveAsTextFile("LRResultPredictedSolution")

