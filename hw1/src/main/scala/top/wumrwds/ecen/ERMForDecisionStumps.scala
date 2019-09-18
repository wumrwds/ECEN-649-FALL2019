package top.wumrwds.ecen

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.sql.functions.{asc, sum}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
  * An efficient implementation of ERM for decision stumps.
  *
  * Note that here I just use Spark as a tool to load data from local csv files and to make some operations on
  * the datasets(as dataframe is quite easy to use in this case).
  *
  * So that means actually I don't utilize any characteristics of parallelism in Spark.
  *
  * @author Minreng Wu
  */
object ERMForDecisionStumps extends StrictLogging {

    final val FEATURE_COL_NAME_PREFIX = "Feature"
    final val LABEL_COL_NAME = "Label"
    final val DISTRIBUTION_COL_NAME = "Distribution"

    /**
      * Calculates the ERM of decision stumps for the given training data set, and returns its corresponding dataset
      * filename, j and θ.
      *
      * @param sparkSession
      * @param originalDataset
      * @param fileName
      * @return
      */
    def calculateERM(sparkSession: SparkSession, originalDataset: Dataset[Row], fileName: String): (String, Double, Double) = {
        // convert column type from string to double
        val dataset = originalDataset.select(originalDataset.columns.map(originalDataset(_).cast(DoubleType)): _*)
                .persist(StorageLevel.MEMORY_AND_DISK)

        // get dimension d and the size of training set m (Suppose that m is within the integer range)
        val d = dataset.columns.filter(_.startsWith("Feature")).size
        val m = dataset.count().toInt

        // f* = ∞
        var fStar = Double.MaxValue
        var thetaStar = Double.MaxValue
        var jStar = Double.MaxValue

        for (j <- 0 until d) {
            // sort df by column Xi,j
            val orderedDataset = dataset.sort(asc(s"$FEATURE_COL_NAME_PREFIX$j"))

            orderedDataset.show(10, false)

            var f = orderedDataset.filter(row => row.getAs[Double](LABEL_COL_NAME) == 1.0)
                    .agg(sum(DISTRIBUTION_COL_NAME))
                    .first().get(0).asInstanceOf[Double]

            // get the array of feature j instances
            val dataArray = orderedDataset.collect()

            // define the functions for retrieving X_i, label and distribution as a double value
            def getDoubleElement(rowIndex: Int, colName: String) = {
                dataArray(rowIndex).getAs[Double](colName)
            }

            def getXi(i: Int, j: Int) = {
                val colName = s"$FEATURE_COL_NAME_PREFIX$j"
                if (0 <= i && i < m) getDoubleElement(i, colName)
                else getDoubleElement(m - 1, colName) + 1
            }

            if (f < fStar) {
                fStar = f
                thetaStar = getXi(0, j) - 1
                jStar = j
            }

            for (i <- 0 until m) {
                f = f - dataArray(i).getAs[Double](LABEL_COL_NAME) * dataArray(i).getAs[Double](DISTRIBUTION_COL_NAME)

                if (f < fStar && getXi(i, j) != getXi(i + 1, j)) {
                    fStar = f
                    thetaStar = (getXi(i, j) + getXi(i + 1, j)) / 2
                    jStar = j
                }
            }
        }

        dataset.unpersist()

        logger.info("+++++ The model output is: F* = {}, j* = {}, θ* = {} +++++", fStar, jStar + 1, thetaStar)

        (fileName, jStar + 1, thetaStar)
    }

    def main(args: Array[String]): Unit = {
        // initialize the spark session
        val sparkSession = SparkSession
                .builder()
                .master("local[2]")
                .appName("Implementation_of_ERM_for_decision_stumps")
                .getOrCreate()

        // loop through each data set
        val results = (0 to 9).map { i =>
            // load original dataset from file system
            val fileName = s"Dataset_number_$i.csv"
            val originalDataset = sparkSession.read.format("csv").option("header", "true")
                    .load(s"dataset/$fileName")

            // call the calculating function
            calculateERM(sparkSession, originalDataset, fileName)
        }

        // convert the result as a dataframe and show it in console
        val resultDf = sparkSession.createDataFrame(results).toDF("fileName", "j*", "θ*")
        resultDf.show(false)

        sparkSession.stop()
    }

}
