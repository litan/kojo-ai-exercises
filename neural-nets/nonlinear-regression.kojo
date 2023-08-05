// #include /nn.kojo
// #include /plot.kojo

cleari()
clearOutput()

val a = 2
val b = 3
val c = 1

val xData0 = Array.tabulate(20)(e => (e + 1).toDouble)
val yData0 = xData0 map (x => a * x * x + b * x + c + random(-5, 5))

val xNormalizer = new StandardScaler()
val yNormalizer = new StandardScaler()

val xData = xNormalizer.fitTransform(xData0)
val yData = yNormalizer.fitTransform(yData0)

val chart = scatterChart("Regression Data", "X", "Y", xData0, yData0)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xDataf = xData.map(_.toFloat)
val yDataf = yData.map(_.toFloat)

val model = new NonlinearModel
model.train(xDataf, yDataf)
val yPreds = model.predict(xDataf)
val yPreds0 = yNormalizer.inverseTransform(yPreds.map(_.toDouble))
addLineToChart(chart, Some("model"), xData0, yPreds0)
drawChart(chart)
model.close()

class NonlinearModel {
    val LEARNING_RATE: Float = 0.1f

    val graph = new Graph
    val tf = Ops.create(graph)
    val session = new Session(graph)

    // truncated normal with shape [a, b]
    def randomw(a: Int, b: Int) =
        tf.math.mul(
            tf.random.truncatedNormal(tf.array(a, b), classOf[TFloat32]),
            tf.constant(0.1f)
        )

    val hidden1 = 15
    val hidden2 = 10
    // Define variables
    val weight = tf.variable(randomw(1, hidden1))
    val bias = tf.variable(tf.zeros(tf.array(hidden1), classOf[TFloat32]))

    val weight2 = tf.variable(randomw(hidden1, hidden2))
    val bias2 = tf.variable(tf.zeros(tf.array(hidden2), classOf[TFloat32]))

    val weight3 = tf.variable(randomw(hidden2, 1))
    val bias3 = tf.variable(tf.zeros(tf.array(1), classOf[TFloat32]))

    def placeholders = {
        // Define placeholders
        val xData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.of(-1, 1)))
        val yData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.of(-1, 1)))
        (xData, yData)
    }

    import scala.util.Using
    import org.tensorflow.Operand

    def modelFunction(xData: Placeholder[TFloat32]): Operand[TFloat32] = {
        // Define the model function
        val mul = tf.linalg.matMul(xData, weight)
        val add = tf.math.add(mul, bias)
        val output1 = tf.nn.relu(add)

        val mul2 = tf.linalg.matMul(output1, weight2)
        val add2 = tf.math.add(mul2, bias2)
        val output2 = tf.nn.relu(add2)

        val mul3 = tf.linalg.matMul(output2, weight3)
        tf.math.add(mul3, bias3)
    }

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
        val N = xValues.length

        val (xData, yData) = placeholders
        val yPredicted = modelFunction(xData)

        // Define loss function MSE
        val sum = tf.math.pow(tf.math.sub(yData, yPredicted), tf.constant(2f))
        val mse = tf.math.div(sum, tf.constant(1f * N))

        // Back-propagate gradients to variables for training
        val optimizer = new GradientDescent(graph, LEARNING_RATE)
        val minimize = optimizer.minimize(mse)

        // Train the model on data
        Using.Manager { use =>
            for (epoch <- 1 to 400) {
                val xTensor = use(TFloat32.tensorOf(
                    Shape.of(N, 1),
                    DataBuffers.of(xValues, true, false)
                ))
                val yTensor = use(TFloat32.tensorOf(
                    Shape.of(N, 1),
                    DataBuffers.of(yValues, true, false)
                ))
                session.runner
                    .addTarget(minimize)
                    .feed(xData.asOutput, xTensor)
                    .feed(yData.asOutput, yTensor)
                    .run()
            }
        }
    }

    def predict(xValues: Array[Float]): Array[Float] = {
        val (xData, yData) = placeholders
        val yPredicted = modelFunction(xData)

        Using.Manager { use =>
            val xTensor = use(TFloat32.tensorOf(
                Shape.of(xValues.length, 1),
                DataBuffers.of(xValues, true, false)
            ))
            val yPredictedTensor = use(session.runner
                .feed(xData, xTensor)
                .fetch(yPredicted)
                .run().get(0).asInstanceOf[TFloat32])

            val predictedY = new Array[Float](xValues.length)
            val predictedYBuffer = yPredictedTensor.asRawTensor().data().asFloats()
            predictedYBuffer.read(predictedY)
            predictedY
        }.get
    }

    def close() {
        session.close()
        graph.close()
    }
}
