import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

class logisticRegressionWithFeatures1 {
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
 var vData = newData.map{case(k,v) => (k,{
   //v.split(" ")
   val arr= v.split(' ').map(a=>(Math.abs(a.toDouble)))
   var i=0
  val list= new ListBuffer[Double]
   var V1=0.0
   var P7=0.0
   var E10=0.0
   var V8=0.0
   var V6=0.0
   
  while(i<arr.length){
   //var value=0.0
   
    if(i%30==19){
      V1=arr(i).toDouble
      list+=V1
    }else if(i%30==6){
      P7=arr(i).toDouble
      list+=Math.pow(P7,3)
      
    }else if(i%30==17){
       E10=arr(i).toDouble
       list+=Math.pow(E10,2)
    }else if(i%30==26){
       V8=arr(i).toDouble
       list+=V8
    }else if(i%30==24){
       V6=arr(i).toDouble
       list+=V6
    }
   
   i+=1
  }
   list.toArray
   })}

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
svmlibConverter("fordTrain.md").saveAsTextFile("trainingDataLogResPlain")
svmlibConverter("fordTest.md").saveAsTextFile("testingDataLogResPlain")


val traindata = MLUtils.loadLibSVMFile(sc, "trainingDataLogResPlain/part-0000*").cache()
val testdata = MLUtils.loadLibSVMFile(sc, "testingDataLogResPlain/part-0000*")
	

val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(traindata)

// Compute raw scores on the test set.
val predictionAndLabels = testdata.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
predictionAndLabels.filter{x=>x._1==x._2}.count().toDouble/ predictionAndLabels.count()
predictionAndLabels.coalesce(1).saveAsTextFile("LRResultPredictedSolution")
}
}
