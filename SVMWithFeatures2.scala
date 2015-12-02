import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV}

object SVMWithFeatures2 {
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
   //var arr=v.split(" ")
  val arr= v.split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  var E7=0.0
  var E9=0.0
  var V1=0.0
  var E8=0.0
  var V2=0.0
  var E11=0.0
  var V5=0.0
  var E4=0.0
  var V4=0.0
  while(i<arr.length){
   var value=0.0
   
    if(i%30==14){
      E7=arr(i).toDouble
    }else if(i%30==16){
      E9=arr(i).toDouble
      value=E7*E9
      list+=value
      
    }else if(i%30==19){
       V1=arr(i).toDouble
       value=E9*V1
       list+=value
    }else if(i%30==15){
       E8=arr(i).toDouble
       
    }else if(i%30==20){
       V2=arr(i).toDouble
       value=E8*V2
       list+=value
    }else if(i%30==18){
       E11=arr(i).toDouble
       
    }else if(i%30==23){
       V5=arr(i).toDouble
       value=E11*V5
       list+=value
    }else if(i%30==11){
      E4=arr(i).toDouble
       
    }else if(i%30==22){
       V4=arr(i).toDouble
       value=E4*V4
       list+=value
    }
  
   i+=1
  }
   list.toArray
   })}
val s=vData.take(1)
s.apply(0)._2.foreach(println)
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
   val arr= v.split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
   var E7=0.0
   var E9=0.0
   var V1=0.0
   var E8=0.0
   var V2=0.0
   var E11=0.0
   var V5=0.0
   var E4=0.0
   var V4=0.0
  while(i<arr.length){
   var value=0.0
   
    if(i%30==14){
      E7=arr(i).toDouble
      
    }else if(i%30==16){
      E9=arr(i).toDouble
      value=E7*E9
      list+=value
      
    }else if(i%30==19){
       V1=arr(i).toDouble
       value=E9*V1
       list+=value
    }else if(i%30==15){
       E8=arr(i).toDouble
       
    }else if(i%30==20){
       V2=arr(i).toDouble
       value=E8*V2
       list+=value
    }else if(i%30==18){
       E11=arr(i).toDouble
       
    }else if(i%30==23){
       V5=arr(i).toDouble
       value=E11*V5
       list+=value
    }else if(i%30==11){
      E4=arr(i).toDouble
       
    }else if(i%30==22){
       V4=arr(i).toDouble
       value=E4*V4
       list+=value
    }
   
   i+=1
  }
   list.toArray
   })}

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


val numIterations = 100
val model = SVMWithSGD.train(TrainData, numIterations,0.01,0.1)

// Compute raw scores on the test set.
val scoreAndLabels = TestData.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

////////////////////////////////////////////
def toBreeze(value:Vector): BV[Double] = new BDV[Double](value.toArray)

def predictPoint(
      dataMatrix: Vector,
      weightMatrix: Vector,
      intercept: Double,
      threshold:Option[Double]) = {
  val brezeWvec= toBreeze(weightMatrix)
  val brezeDatavec= toBreeze(dataMatrix)
   val margin = brezeWvec.dot(brezeDatavec) + intercept
   
    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  
  }
////////////////////////////////////////////////////////////

val bc_model=sc.broadcast(model)
val scoreAndPredict = TestData.map { point =>
  val score = predictPoint(point.features,bc_model.value.weights,bc_model.value.intercept,bc_model.value.getThreshold)
  (score, point.label)
}
scoreAndPredict.foreach(println)
scoreAndPredict.coalesce(1).saveAsTextFile("SVM2Predict")
val accuracy = 1.0 * scoreAndPredict.filter(x => x._1 == x._2).count() / TestData.count()
// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC = " + auROC)





  }
  
}
  