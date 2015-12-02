import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object StatsCalc {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("StatsCalc").setMaster("local[*]")
    val sc = new SparkContext(conf)
    
    val resultsFile = sc.textFile(args(0))
    
    //get tp, fn, fp, tn
    val basicBlocks = resultsFile.map { 
      x => val values = x.replace("(", "").replace(")", "").split(",")
      (values(0).toDouble, values(1).toDouble) }
    .map { case(p, o) => (2*p.toInt+o.toInt, 1.0) }
    .reduceByKey(_+_)
    .sortBy(_._1)
    .collect
    
    val tp = basicBlocks(0)._2
    val fp = basicBlocks(1)._2
    val fn = basicBlocks(2)._2
    val tn = basicBlocks(3)._2
    
    val tpr = tp/(tp+fn)  //true +ve rate or recall
    val fpr = fp/(fp+tn)  //false +ve rate
    val precision = tp/(tp+fp)
    
    val acc = (tp+tn)/(tp+fp+fn+tn)  //accuracy
    val aucroc = (tpr - fpr + 1)/2
    val f1 = (2 * precision * tpr)/(precision + tpr)
    println("------"+args(0).toString+" Results--------------\n")
    println("ACCURACY="+acc)
    println("AUC ROC="+aucroc)
    println("F1 SCORE="+f1+"\n")
    println(tp + " " + fp + " " + fn + " " +tn)
  }
}
