package top.wumrwds.ecen

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.sql.functions.{asc, sum}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
  * @author Minreng Wu
  */
object ERMForDecisionStumps extends StrictLogging {

    def calculateERM(sparkSession: SparkSession, originalDataset: Dataset[Row], fileName: String): (String, Double, Double) = {
        // convert column type from string to double
        val dataset = originalDataset.select(originalDataset.columns.map(originalDataset(_).cast(DoubleType)): _*)
                .persist(StorageLevel.MEMORY_AND_DISK)

        val featureColNamePrefix = "Feature"
        val labelColName = "Label"
        val distributionColName = "Distribution"

        // get dimension d and the size of training set m (Suppose that m is within the integer range)
        val d = dataset.columns.filter(_.startsWith("Feature")).size
        val m = dataset.count().toInt

        // f* = ∞
        var fStar = Double.MaxValue
        var thetaStar = Double.MaxValue
        var jStar = Double.MaxValue

        for (j <- 0 until d) {
            val orderedDataset = dataset.sort(asc(s"$featureColNamePrefix$j"))

            orderedDataset.show(10, false)

            var f = orderedDataset.filter(row => row.getAs[Double](labelColName) == 1.0)
                    .agg(sum(distributionColName))
                    .first().get(0).asInstanceOf[Double]

            // get the array of feature j instances
            val dataArray = orderedDataset.collect()

            def getDoubleElement(rowIndex: Int, colName: String) = {
                dataArray(rowIndex).getAs[Double](colName)
            }

            def getXi(i: Int, j: Int) = {
                val colName = s"$featureColNamePrefix$j"
                if (0 <= i && i < m) getDoubleElement(i, colName)
                else getDoubleElement(m - 1, colName) + 1
            }

            if (f < fStar) {
                fStar = f
                thetaStar = getXi(0, j) - 1
                jStar = j
            }

            for (i <- 0 until m) {
                f = f - dataArray(i).getAs[Double](labelColName) * dataArray(i).getAs[Double](distributionColName)

                if (f < fStar && getXi(i, j) != getXi(i + 1, j)) {
                    fStar = f
                    thetaStar = (getXi(i, j) + getXi(i + 1, j)) / 2
                    jStar = j
                }
            }
        }

        dataset.unpersist()

        logger.info("+++++ The model output is: F* = {}, j* = {}, θ* = {} +++++", fStar, jStar + 1, thetaStar)

        (fileName, jStar + 1, fStar)
    }

    def main(args: Array[String]): Unit = {
        // init spark session
        val sparkSession = SparkSession
                .builder()
                .master("local[2]")
                .appName("Implementation_of_ERM_for_decision_stumps")
                .getOrCreate()

        val results = (0 to 9).map { i =>
            // load original dataset from file system
            val fileName = s"Dataset_number_$i.csv"
            val originalDataset = sparkSession.read.format("csv").option("header", "true")
                    .load(s"dataset/$fileName")

            calculateERM(sparkSession, originalDataset, fileName)
        }

        val resultDf = sparkSession.createDataFrame(results).toDF("fileName", "j*", "θ*")

        resultDf.show(false)

        sparkSession.stop()
    }

}
