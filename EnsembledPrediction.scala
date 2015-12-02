
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


class EnsembledPrediction {
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("prep").setMaster("local")
    val sc = new SparkContext(conf)
  
    val naiveModelPrediction=sc.textFile("NaivePredict/part-0000*").map { x => (x.split(",")(0),x.split(",")(1)) }
    val rddNaive=naiveModelPrediction.zipWithIndex().map(x=>(x._2,x._1))
    /*val svm1ModelPrediction=sc.textFile("SVM1Predict/part-00000").map { x => (x.split(",")(0)) }
    val rddSVM1=svm1ModelPrediction.zipWithIndex().map(x=>(x._2,(x._1)))*/
    val svm2ModelPrediction=sc.textFile("SVM2Predict/part-00000").map { x => (x.split(",")(0)) }
    val rddSVM2=svm2ModelPrediction.zipWithIndex().map(x=>(x._2,(x._1)))
    val LR1ModelPrediction=sc.textFile("LRResultPredictedSolution/part-00000").map { x => (x.split(",")(0)) }
    val rddLR1=LR1ModelPrediction.zipWithIndex().map(x=>(x._2,(x._1)))
    val LR2ModelPrediction=sc.textFile("LRResultPredictedSolution/part-00000").map { x => (x.split(",")(0)) }
    val rddLR2=LR2ModelPrediction.zipWithIndex().map(x=>(x._2,(x._1)))
    
    
   val join1=rddSVM2.join(rddNaive)
   val s= join1.map(x=>(x._1,x._2._1,x._2._2._1,x._2._2._2))
   s.sortBy(x=>x._1).coalesce(1).saveAsTextFile("Join")
   val rddLoad=sc.textFile("Join/part-00000").map { x =>var v1=x.replaceAll("\\(", ""); var v2=v1.replaceAll("\\)", "");v2}
   val rddJoin=rddLoad.map { x => (x.split(",")(0).toLong,(x.split(",")(1),x.split(",")(2),x.split(",")(3)) )}
   // rddJoin.foreach(println)
   val join2= rddLR1.join(rddJoin)
   val v= join2.map(x=>(x._1,x._2._1,x._2._2._1,x._2._2._2,x._2._2._3))
   v.sortBy(x=>x._1).coalesce(1).saveAsTextFile("Join1")
   
   val rddLoad1=sc.textFile("Join1/part-00000").map { x =>var v1=x.replaceAll("\\(", ""); var v2=v1.replaceAll("\\)", "");v2}
   val rddJoin1=rddLoad1.map { x => (x.split(",")(0).toLong,(x.split(",")(1),x.split(",")(2),x.split(",")(3),x.split(",")(4)) )}
   val join3= rddLR2.join(rddJoin1)
   val v1= join3.map(x=>(x._1,x._2._1,x._2._2._1,x._2._2._2,x._2._2._3,x._2._2._4))
   v1.sortBy(x=>x._1).coalesce(1).saveAsTextFile("Join2")
   
   val finalrddLoad=sc.textFile("Join2/part-00000").map { x =>var v1=x.replaceAll("\\(", ""); var v2=v1.replaceAll("\\)", "");v2}
   val finalRdd=finalrddLoad.map { x => (x.split(",")(5),(x.split(",")(1),x.split(",")(2),x.split(",")(3),x.split(",")(4)) )}
   
   finalRdd.coalesce(1).saveAsTextFile("FinalPrediction")
   
   val data=sc.textFile("FinalPrediction/part-00000").map { x =>var v1=x.replaceAll("\\(", ""); var v2=v1.replaceAll("\\)", "");v2}
   val rddCounter=data.map { x => ({
     var counter1=x.split(",")(1);
     var counter2=x.split(",")(2);
     var counter3=x.split(",")(3);
     var counter4=x.split(",")(4);
     var CountOf0=0
     var CountOf1=0
     var value=0.0
     if(counter1.toDouble==0.0){
       CountOf0+=1
     }else if(counter1.toDouble==1.0){
       CountOf1+=1
     }
     if(counter2.toDouble==0.0){
       CountOf0+=1
     }else if(counter2.toDouble==1.0){
       CountOf1+=1
     }
     if(counter3.toDouble==0.0){
       CountOf0+=1
     }else if(counter3.toDouble==1.0){
       CountOf1+=1
     }
     if(counter4.toDouble==0.0){
       CountOf0+=1
     }else if(counter4.toDouble==1.0){
       CountOf1+=1
     }
     if(CountOf1>=CountOf0){
       value=1.0
     }
     value
     },x.split(",")(0).toDouble )}
   val accuracy = 1.0 * rddCounter.filter(x => x._1 == x._2).count() / rddCounter.count()
  rddCounter.coalesce(1).saveAsTextFile("EnsembledResult")
   
  }
  
  
}