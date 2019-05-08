import java.io
import org.jsoup.Jsoup
import org.jsoup.nodes._
import org.jsoup.select._
import org.apache.spark.mllib.util._
import org.apache.spark.mllib.tree._ 
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

def processData (allFiles: RDD[java.io.File]) = {
    val toFlatData = allFiles.map(file => {
        val doc = Jsoup.parse(file, "UTF-8")
        val dataDiv = doc.getElementsByTag("div").asScala.map(div => {
            val docTitle = doc.title.replaceAll("[,.;!?(){}\\[\\]<>%]", "")
            val docLinks = doc.select("a[href]").size
            val docWords = doc.body().text.split("[\\W]+").size
            val docDensity = if (docLinks > 0 && docWords > 0) {docWords / docLinks.toFloat} else {0}
            val docParag = doc.select("p").size
            val docImages = doc.select("img").size
            val docChildSize = doc.childNodeSize
            val docDiv = doc.getElementsByTag("div").size
            val divLinks = div.select("a[href]").size
            val divWords = div.text.split("[\\W]+").size
            val divDensity = if (divLinks > 0 && divWords > 0) {divWords / divLinks.toFloat} else {0}
            val divParag = div.select("p").size
            val divImages = div.select("img").size
            val divChildSize = div.childNodeSize
            val divClass = div.className
            val goodBad = if (divClass.endsWith("-XXXgood")) {1} else {0}
            (docTitle, docLinks, docWords, docDensity, docParag, docImages, docChildSize, docDiv, 
                divLinks, divWords, divDensity, divParag, divImages, divChildSize, divClass, goodBad)
        })
        (dataDiv)
    })
    toFlatData.flatMap(x => x.map(s =>s))
}

def createFeaturesVector (inputRDD: 
    RDD[(String, Int, Int, Float, Int, Int, Int, Int, Int, Int, Float, Int, Int, Int, String, Int)]) = {
        val vector01 = inputRDD.map(x => (LabeledPoint(x._16.toDouble, 
            Vectors.dense(x._9.toDouble, x._10.toDouble, x._11.toDouble,
                x._12.toDouble, x._13.toDouble, x._14.toDouble))))
        val vector02 = inputRDD.map(x => (LabeledPoint(x._16.toDouble, 
            Vectors.dense(x._2.toDouble, x._3.toDouble, x._4.toDouble,
                x._5.toDouble, x._6.toDouble, x._7.toDouble, x._8.toDouble,
                x._9.toDouble, x._10.toDouble, x._11.toDouble,
                x._12.toDouble, x._13.toDouble, x._14.toDouble))))
        val vector03 = inputRDD.map(x => (LabeledPoint(x._16.toDouble, 
            Vectors.dense(x._2.toDouble, x._3.toDouble, x._5.toDouble, 
                x._6.toDouble, x._9.toDouble, x._10.toDouble,
                x._12.toDouble, x._13.toDouble))))
        val vector04 = inputRDD.map(x => (LabeledPoint(x._16.toDouble, 
            Vectors.dense(x._4.toDouble, x._5.toDouble, 
                x._6.toDouble, x._8.toDouble, x._11.toDouble, 
                x._12.toDouble, x._13.toDouble))))
        val vector05 = inputRDD.map(x => (LabeledPoint(x._16.toDouble, 
            Vectors.dense(x._2.toDouble, x._3.toDouble, x._4.toDouble,
                x._5.toDouble, x._6.toDouble, x._7.toDouble, x._8.toDouble,
                x._9.toDouble, x._11.toDouble, x._12.toDouble,
                x._13.toDouble, x._14.toDouble))))
    Array(vector01, vector02, vector03, vector04, vector05)
}

// Capture the entire path for all files in the current folder ending with "html" extension 
// It results in an array which will be parallelized to serve as an input in the def processData.
val allFiles = sc.parallelize(new io.File("./").listFiles.filter(_.getName.endsWith(".html")).map(_.getCanonicalFile))
val Array(allTrainFiles, allTestFiles, allValFiles) = allFiles.randomSplit(Array(0.5,0.3,0.2))
val allTrainData = processData(allTrainFiles)
val allTestData = processData(allTestFiles)
val allValData = processData(allValFiles)
println("\n\n\n>>>>>>> Total Files: " + allFiles.count)
println(">>>>>>> Number Train Files: " + allTrainFiles.count)
println(">>>>>>> Number Test Files: " + allTestFiles.count)
println(">>>>>>> Number Val Files: " + allValFiles.count)
println(">>>>>>> Number Train Lines: " + allTrainData.count)
println(">>>>>>> Number Test Lines: " + allTestData.count)
println(">>>>>>> Number Val Lines: " + allValData.count)
val featuresVectorTrain = createFeaturesVector(allTrainData)
val featuresVectorTest = createFeaturesVector(allTestData)
val featuresVectorVal = createFeaturesVector(allValData)

/// ------------------- Testing Scenarios
println("Working on Files From: " + new java.io.File(".").getCanonicalPath)
for (x <- 0 to featuresVectorTrain.size-1) {
    val trainingData = featuresVectorTrain(x)
    val testData = featuresVectorTest(x)
    val valData = featuresVectorVal(x)

    trainingData.persist()
    valData.persist()

    val k = if (allFiles.count > 100) {7} else {2}
    val cvData = MLUtils.kFold(trainingData, k, 0)
    val numClasses = 2
    val catFeature = Map[Int, Int]()

    // Evaluation to find the best model
    val modelTuningDT = for (
        (trainingData, valData) <- cvData; 
        impurity <- Array("gini", "entropy"); 
        depth <- Array(5, 10, 20); 
        bins <- Array(50, 100, 200, 300)) yield {
        val model = DecisionTree.trainClassifier(trainingData, numClasses, catFeature, impurity, depth, bins)
        val predictionsAndLabels = valData.map(example => (model.predict(example.features), example.label))
        val areaUnderPR = new BinaryClassificationMetrics(predictionsAndLabels).areaUnderPR
        ((impurity, depth, bins), areaUnderPR)
    }

    val bestDTImpurity = modelTuningDT.maxBy(_._2)._1._1
    val bestDTDepth = modelTuningDT.maxBy(_._2)._1._2
    val bestDTBins = modelTuningDT.maxBy(_._2)._1._3

    val modelTuningRF = for ((trainingData, valData) <- cvData;
        trees <- Array(5, 10, 20); 
        impurity <- Array("gini", "entropy"); 
        depth <- Array(5, 10, 20); 
        bins <- Array(50, 100, 200, 300)) yield {
        val model = RandomForest.trainClassifier(trainingData, numClasses, catFeature, trees, "auto", impurity, depth, bins)
        val predictionsAndLabels = valData.map(example => (model.predict(example.features), example.label))
        val areaUnderPR = new BinaryClassificationMetrics(predictionsAndLabels).areaUnderPR
        ((trees, impurity, depth, bins), areaUnderPR)
    }

    val bestNuTrees = modelTuningRF.maxBy(_._2)._1._1
    val bestRFImpurity = modelTuningRF.maxBy(_._2)._1._2
    val bestRFDepth = modelTuningRF.maxBy(_._2)._1._3
    val bestRFBins = modelTuningRF.maxBy(_._2)._1._4

    // Based on Model Tuning, training the classifier
    val modelDT = DecisionTree.trainClassifier(trainingData, numClasses, catFeature, bestDTImpurity, bestDTDepth, bestDTBins)
    val modelRF = RandomForest.trainClassifier(trainingData, numClasses, catFeature, bestNuTrees, "auto", bestRFImpurity, bestRFDepth, bestRFBins)
    trainingData.unpersist()
    valData.unpersist()
    testData.persist()

    val predictionAndLabelsDT = testData.map { case LabeledPoint(label, features) =>
        val prediction = modelDT.predict(features)
        (prediction, label)
    }

    val predictionAndLabelsRF = testData.map { case LabeledPoint(label, features) =>
        val prediction = modelRF.predict(features)
        (prediction, label)
    }

    testData.unpersist()
    val metricsDT = new BinaryClassificationMetrics(predictionAndLabelsDT)
    val metricsRF = new BinaryClassificationMetrics(predictionAndLabelsRF)

    // Metrics
    println("\n\n====== Printing Results for Scenario " + (x.toInt + 1))
    println("Precision DT: " + metricsDT.precisionByThreshold.max._2)
    println("Precision RF: " + metricsRF.precisionByThreshold.max._2)
    println("Recall DT: " + metricsDT.recallByThreshold.max._2)
    println("Recall RF: " + metricsRF.recallByThreshold.max._2)
    println("F-Measure DT: " + metricsDT.fMeasureByThreshold.max._2)
    println("F-Measure RF: " + metricsRF.fMeasureByThreshold.max._2)
    println("Area under P-R Curve DT: " + metricsDT.areaUnderPR)
    println("Area under P-R Curve RF: " + metricsRF.areaUnderPR)
}