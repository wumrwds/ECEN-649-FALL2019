package top.wumrwds.ecen

import org.apache.spark.sql.SparkSession

/**
  * @author Minreng Wu
  */
object ERMForDecisionStumps {

    def main(args: Array[String]): Unit = {
        // init spark session
        val sparkSession = SparkSession
                .builder()
                .master("local[2]")
                .appName("Implementation_of_ERM_for_decision_stumps")
                .getOrCreate()

        val dataTest = sparkSession.read.format("csv").option("header", "true")
                .load("dataset/Dataset_number_0.csv")

        dataTest.show(1000, false)

        sparkSession.stop()
    }

}
