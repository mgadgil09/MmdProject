
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
//import org.apache.spark.mllib.feature.MDLPDiscretizer
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

class NaiveBayes {
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
  val parsedData = parse.map { line =>
  val parts = line.split(',')
  val s= parts(1).split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  var prevVal=0.0
  while(i<s.length){
    var value=0.0
    
    if(i%30==16){
      value=s(i).toDouble*4.4185
    }
    else if(i%30==29){
       value=s(i).toDouble*0.1494
    }else if(i%30==12){
       value=s(i).toDouble
    }
    /*if(i%30!=7 || i%30!=25 || i%30!=27){
      value=s(i).toDouble*/
    
    list+=value
    /*list+=1.0/value
    list+=Math.pow(value,2)
    list+=Math.pow(value,3)
    if(i%30!=0){
       list+=prevVal*value
    }*/
    
   // }
    prevVal=value
    i+=1
  }
  LabeledPoint(parts(0).split(" ")(2).toDouble, Vectors.dense(list.toArray))
}
  
  val model = NaiveBayes.train(parsedData, lambda = 1.0, modelType = "multinomial")
  
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
  
  val testParsedData = testparse.map { line =>
  val testparts = line.split(',')
  val s= testparts(1).split(' ').map(a=>(Math.abs(a.toDouble)))
  var i=0
  val list= new ListBuffer[Double]
  var prevVal=0.0
  while(i<s.length){
    var value=0.0
    if(i%30==16){
      value=s(i).toDouble*4.4185
    }
    else if(i%30==29){
       value=s(i).toDouble*0.1494
    }else if(i%30==12){
       value=s(i).toDouble
    }
    /*if(i%30!=7 || i%30!=25 || i%30!=27){
      value=s(i).toDouble*/
   
    list+=value
    /*list+=1.0/value
    list+=Math.pow(value,2)
    list+=Math.pow(value,3)
    if(i%30!=0){
       list+=prevVal*value
    }
    
     }*/
    prevVal=value
    i+=1
  }
  LabeledPoint(testparts(0).split(" ")(2).toDouble, Vectors.dense(list.toArray))
}
  
  
  val predictionAndLabel = testParsedData.map(p => (model.predict(p.features),p.label))
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testParsedData.count()
  //predictionAndLabel.foreach {println}
  
   
  }  
}