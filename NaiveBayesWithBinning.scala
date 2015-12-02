
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

class NaiveBayesWithBinning {
  def main(args: Array[String]){
  val conf = new SparkConf().setAppName("prep").setMaster("local")
    val sc = new SparkContext(conf)
  
  
  val data = sc.textFile("fordTrain.md")
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
   //var arr=v.split(" ")
  val arr= v.split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  while(i<arr.length){
   var value=0.0
    value=arr(i).toDouble
    list+=value
   // list+=1.0/value
    //list+=Math.pow(value,2)
    //list+=Math.pow(value,3)  
   i+=1
  }
   list.toArray
   })}

val header = parse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

//val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)
var joined = vData.map{case(k,v) => var newV = for(i <- 0 until v.length)yield{(i+1).toString+":"+Math.abs(Math.getExponent(v(i)).toDouble)+" "}
(k,newV.mkString.trim).toString.trim}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}
joined.saveAsTextFile("TrainNaiveResult")

val TrainData = MLUtils.loadLibSVMFile(sc, "TrainNaiveResult/part-0000*")
val discretizedData = TrainData.map { lp =>
  LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 15).floor } ) )
}
  val model = NaiveBayes.train(discretizedData, lambda = 1.0, modelType = "multinomial")
  
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
  while(i<arr.length){
   var value=0.0
    value=arr(i).toDouble
      list+=value
     // list+=1.0/value
     // list+=Math.pow(value,2)
     // list+=Math.pow(value,3)
   i+=1
  }
   list.toArray
   })}

val testheader = testparse.take(1).map{line =>
      val parts = line.split(",")(0).split(" ")(2)
      (parts,line.split(",")(1).trim)}

//val noLabel = header.map{case(k,v) => v.split(" ")}.flatMap(line => line)
var testJoined = vTestData.map{case(k,v) => var newV = for(i <- 0 until v.length)yield{(i+1).toString+":"+Math.abs(Math.getExponent(v(i)).toDouble)+" "}
(k,newV.mkString.trim).toString.trim}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}
testJoined.saveAsTextFile("TestNaiveResult")
  
val TestData = MLUtils.loadLibSVMFile(sc, "TestNaiveResult/part-0000*")
val discretizedData1 = TestData.map { lp =>
  LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 15).floor } ) )
}
  val predictionAndLabel = discretizedData1.map(p => (model.predict(p.features),p.label))
  
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / TestData.count()
  predictionAndLabel.saveAsTextFile("result")
  
   
  }  
}