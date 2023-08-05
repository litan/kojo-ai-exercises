import org.tensorflow.types.TFloat32
import org.tensorflow.{ SavedModelBundle, Tensor }
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers

import java.io.File
import java.util.HashMap
import java.awt.image.BufferedImage

import net.kogics.kojo.util.Utils

def imgGrayToTensorF(image: BufferedImage): TFloat32 = {
    import java.nio.ByteBuffer
    val h = image.getHeight
    val w = image.getWidth
    val imgBuffer = ByteBuffer.allocate(h * w * 1 * 4)

    for (y <- 0 until h) {
        for (x <- 0 until w) {
            val pixel = image.getRGB(x, y)
            val gray = (pixel >> 16) & 0xff
            imgBuffer.putFloat(gray.toFloat / 255f)
        }
    }
    imgBuffer.flip()
    val shape = Shape.of(1, image.getHeight * image.getWidth)
    val db = DataBuffers.of(imgBuffer).asFloats()
    val t2 = TFloat32.tensorOf(shape, db)
    t2
}

cleari()

val scriptDir = Utils.kojoCtx.baseDir
val mnistModel = s"$scriptDir/mnist_model_967"

require(
    new File(mnistModel).exists,
    s"Cannot find MNIST model dir: ${mnistModel}")

val mnistNet = SavedModelBundle.load(mnistModel)

def predict(inputTensor: TFloat32): Int = {
    def argMax(xs: ArrayBuffer[Float]): Int = {
        val (_, maxIdx) = xs.zipWithIndex.maxBy { case (x, i) => x }
        maxIdx
    }
    val args = new HashMap[String, Tensor]()
    args.put("input_1", inputTensor)
    val out = mnistNet.call(args).get("softmax").get.asInstanceOf[TFloat32]
    val data = out.asRawTensor.data.asFloats
    val ab = new ArrayBuffer[Float]
    for (n <- 0 until data.size.toInt) ab.append(data.getFloat(n))
    val result = ab // .toArray
    out.close()
    argMax(result)
}

def makePredictionPic(filename: String, expected: Int): Picture = {
    val inputImage = Utils.loadBufImage(filename)
    val pic = Picture.image(inputImage)
    val inputTensor = imgGrayToTensorF(inputImage)
    val prediction = predict(inputTensor)
    val clr = if (prediction == expected) green else red
    val pic2 = Picture.text(prediction, 30).withPenColor(clr)
    picRowCentered(pic, Picture.hgap(20), pic2)
}

val pics = for (i <- 0 to 9) yield {
    makePredictionPic(s"test/$i.png", i)
}

val pics2 = picCol(pics)
drawCentered(pics2)

mnistNet.close()

//cleari()
//val pic = makePredictionPic("test/3.png", 1)
//drawCentered(pic)
