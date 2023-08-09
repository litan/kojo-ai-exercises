import java.io.IOException

import scala.util.Using

import org.tensorflow._
import org.tensorflow.framework.optimizers.GradientDescent
import org.tensorflow.model.examples.datasets.mnist.MnistDataset
import org.tensorflow.ndarray.ByteNdArray
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

val TRAINING_IMAGES_ARCHIVE = "mnist/train-images-idx3-ubyte.gz"
val TRAINING_LABELS_ARCHIVE = "mnist/train-labels-idx1-ubyte.gz"
val TEST_IMAGES_ARCHIVE = "mnist/t10k-images-idx3-ubyte.gz"
val TEST_LABELS_ARCHIVE = "mnist/t10k-labels-idx1-ubyte.gz"
val VALIDATION_SIZE = 0
val TRAINING_BATCH_SIZE = 100
val LEARNING_RATE = 0.2f

def preprocessImages(rawImages: ByteNdArray) = {
    val tf = Ops.create
    // Flatten images in a single dimension and normalize their pixels as floats.
    val imageSize = rawImages.get(0).shape.size
    tf.math
        .div(
            tf.reshape(
                tf.dtypes.cast(tf.constant(rawImages), classOf[TFloat32]),
                tf.array(-1L, imageSize)
            ),
            tf.constant(255.0f)
        )
        .asTensor
}

def preprocessLabels(rawLabels: ByteNdArray) = {
    val tf = Ops.create
    // Map labels to one hot vectors where only the expected predictions as a value of 1.0
    tf.oneHot(
        tf.constant(rawLabels),
        tf.constant(MnistDataset.NUM_CLASSES),
        tf.constant(1.0f),
        tf.constant(0.0f)
    ).asTensor
}

val dataset = MnistDataset.create(
    VALIDATION_SIZE, TRAINING_IMAGES_ARCHIVE, TRAINING_LABELS_ARCHIVE,
    TEST_IMAGES_ARCHIVE, TEST_LABELS_ARCHIVE
)

val model = new MnistModel()
model.train()
model.test()
model.save()
model.close()

class MnistModel {
    val graph = new Graph()
    val tf = Ops.create(graph)

    // Create placeholders and variables, which should fit batches of an unknown number of images
    val images = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.of(-1, dataset.imageSize)))
    val labels = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.of(-1, MnistDataset.NUM_CLASSES)))

    // Create weights with an initial value of 0
    val weightShape = tf.array(dataset.imageSize, MnistDataset.NUM_CLASSES)
    val weights = tf.variable(tf.zeros(weightShape, classOf[TFloat32]))

    // Create biases with an initial value of 0
    val biasShape = tf.array(MnistDataset.NUM_CLASSES)
    val biases = tf.variable(tf.zeros(biasShape, classOf[TFloat32]))

    // Predict the class of each image in the batch and compute the loss
    val softmax =
        tf.nn.softmax(
            tf.math.add(
                tf.linalg.matMul(images, weights),
                biases
            )
        )

    val crossEntropy =
        tf.math.mean(
            tf.math.neg(
                tf.reduceSum(
                    tf.math.mul(labels, tf.math.log(softmax)),
                    tf.array(1)
                )
            ),
            tf.array(0)
        )

    // Back-propagate gradients to variables for training
    val optimizer = new GradientDescent(graph, LEARNING_RATE)
    val minimize = optimizer.minimize(crossEntropy)

    // Compute the accuracy of the model
    val predicted = tf.math.argMax(softmax, tf.constant(1))
    val expected = tf.math.argMax(labels, tf.constant(1))
    val accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), classOf[TFloat32]), tf.array(0))

    val session = new Session(graph)

    def train() {
        // Train the model
        import scala.jdk.CollectionConverters._
        for (trainingBatch <- dataset.trainingBatches(TRAINING_BATCH_SIZE).asScala) {
            Using.Manager { use =>
                val batchImages = use(preprocessImages(trainingBatch.images))
                val batchLabels = use(preprocessLabels(trainingBatch.labels))
                session.runner
                    .addTarget(minimize)
                    .feed(images, batchImages)
                    .feed(labels, batchLabels)
                    .run()
            }
        }
    }

    def test() {
        Using.Manager { use =>
            // Test the model
            val testBatch = dataset.testBatch
            val testImages = use(preprocessImages(testBatch.images))
            val testLabels = use(preprocessLabels(testBatch.labels))
            val accuracyValue = use(
                session.runner
                    .fetch(accuracy)
                    .feed(images.asOutput, testImages)
                    .feed(labels.asOutput, testLabels)
                    .run
                    .get(0)
                    .asInstanceOf[TFloat32]
            )
            System.out.println("Accuracy: " + accuracyValue.getFloat())
        }
    }

    def save() {
        val sig = Signature.builder.key("serving_default").input("input_1", images).output("softmax", softmax).build
        val sf = new SessionFunction(sig, session)
        try {
            SavedModelBundle.exporter(s"${kojoCtx.baseDir}/mnist_model").withFunction(sf).`export`()
            System.out.println("Model saved")
        }
        catch {
            case e: IOException =>
                System.out.println(e.getMessage)
        }
    }

    def close() {
        session.close()
        graph.close()
    }
}
