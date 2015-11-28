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
					(i+1).toString+":"+v(i)+" "
					
					}
					(k,newV.mkString.trim).toString.trim
		}.map{line => line.substring(1,line.length-2)}.map{line => line.split(",")(0)+" "+line.split(",")(1).trim}


}
}
